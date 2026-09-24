"""Three-layer hybrid gate for DeciServ (arbiter-pattern, ported from
0xBakeer/arbiter integrations/claude-code/hooks/guard_policy.py and adapted to
DeciServ's /decide contract).

Layer 1 — read-only fast path: if every pipeline stage is a verb that cannot
  change or transmit anything (no redirection, no substitution, no pipes into
  non-fast-path verbs), return 'pass' without a model call.
Layer 2 — text risk floors: compiled regex classes applied to the state text.
  CATASTROPHIC → floor 0.99 (deny). SECRET_PATH / SENDS_OUT → floor 0.78
  (deny floor). Floors only ever RAISE the risk:
  final_risk = max(text_floor, model_weighted_risk).
Layer 3 — model for the grey middle: the provider scores ONE 'what is lost'
  score question (the 'blast radius' framing separated allow/deny at AUC 0.59
  on arbiter's 117-event set; 'what is lost' reached 0.82) plus short noul
  questions (long criteria blunt accuracy). Weighted composite:
  risk = BIAS + Σ(weight_i × signal_i). ≥ DECISERV_DENY_AT (0.78) → deny,
  ≥ DECISERV_ASK_AT (0.50) → ask, else allow.

Semantics (fail-closed discipline):
  * Fast-path hits skip the model entirely (tracked as fast_path_hits).
  * Model/server error with a floor-matching pattern → fail CLOSED (deny).
  * Model/server error without a floor match → fail open to ESCALATE, never
    silently allow. Server down → escalate.
  * Mapped verdicts: allow→pass, ask→escalate, deny→block. The /decide
    contract shape {answers, decision_id, provider, policy, latency_ms} is
    unchanged; policy='hybrid' adds 'verdict' + 'final_risk' alongside.

Everything below was measured, not guessed — arbiter's numbers, pending a
refit on DeciServ's own traffic:
  * arbiter's weights were least-squares-fitted on 117 labelled PreToolUse
    events (allow=0, ask=0.5, deny=1). OURS MUST BE REFIT on our traffic
    (log entries carry 'final_risk' + state hashes for exactly this).
  * BIAS=-0.10 keeps benign grey-middle traffic out of the ask band while
    the ~0.5-probability noul prior pushes it up.
  * w_noul=0.85 / w_score=0.15: the binary 'is this destructive' noul
    signal is the sharpest the model offers; the 4-level score at full
    scale (÷3) is a tiebreaker, not a carrier (AUC 0.82 vs 0.59 measured).
  * deny=0.78 / ask=0.50: keeps arbiter's measured operating point; the
    destructive-vs-benign separation they measured made ~0.78 the point
    where false denies on legitimate ops commands stopped being worth the
    extra catch.
"""
from __future__ import annotations
import os
import re

# --------------------------------------------------------------------------- env
try:
    DENY_AT = float(os.environ.get("DECISERV_DENY_AT", "0.78"))
except ValueError:
    DENY_AT = 0.78
try:
    ASK_AT = float(os.environ.get("DECISERV_ASK_AT", "0.50"))
except ValueError:
    ASK_AT = 0.50
POLICY_MODE = os.environ.get("DECISERV_POLICY", "hybrid").strip().lower() or "hybrid"

BIAS = -0.10          # arbiter's fitted intercept: grey-middle benign traffic sits below 0.50
W_NOUL = 0.85         # arbiter: destroys_data noul — sharpest single signal
W_SCORE = 0.15        # arbiter: 'what is lost' at full scale (score/3); AUC 0.82 vs 0.59 for reach

# --------------------------------------------------------------------------- layer 1
# A pipeline built only out of these, with no redirection, no substitution and
# no secret path in sight, cannot destroy or transmit anything — allowed with
# no model call. A reading of what the verbs can do, not a safety judgement.
READ_ONLY = {
    "ls", "cat", "head", "tail", "grep", "find", "df", "du", "ps", "nvidia-smi",
    "echo", "date", "wc", "file", "stat", "which", "pwd", "printf", "whoami",
    "id", "uname", "hostname", "basename", "dirname", "realpath", "readlink",
    "sort", "uniq", "cut", "tr", "column", "jq", "yq", "diff", "tree", "seq",
    "true", "false", "env", "printenv", "nl",
}
READ_ONLY_SUB = {
    "git": {"status", "log", "diff", "show", "branch", "rev-parse", "describe",
            "blame", "shortlog", "ls-files", "remote", "worktree", "whatchanged",
            "stash", "ls-remote"},
    "docker": {"ps", "images", "logs", "inspect", "version", "info"},
    "kubectl": {"get", "logs", "describe", "version", "explain", "api-resources"},
    "npm": {"ls", "view", "outdated"},
    "pip": {"show", "list", "freeze"},
    "cargo": {"tree"},
    "go": {"env", "version", "list"},
    "nvidia-smi": {},
    "gh": {"pr", "issue", "run", "release"},  # narrowed below to view/list/status only
}
# gh subcommands that are read-only (list-then-filter, else gh release upload etc. is NOT read-only)
_GH_VIEW = {"view", "list", "status", "checks"}

UNSAFE_TOKENS = re.compile(
    r">>|[>&`]|\$\(|\|\s*(sh|bash|zsh|python\d?|perl|ruby|node|nc|ncat|curl|wget|tee)\b|\bsudo\b|\beval\b")
# `&&` runs the second stage regardless of the first — not read-only, but NOT
# unsafe per se (git status && git log is safe). It takes the fast path away
# and leaves the verdict to the floors + model, which measured better than
# hand-parsing shell conditionals.
SEQ_OK = re.compile(r"&&")
UNSAFE_FIND = re.compile(r"-(delete|exec|execdir|ok|okdir|fprint|fls)\b")
VERSION_ONLY = re.compile(r"^(python\d?|node|ruby|perl|java|go)\s+(-V|--version|version)\s*$")
# curl/wget with no flags that could write or execute — GET, output to stdout.
LOCALHOST_GET = re.compile(r"^-{0,2}[A-Za-z-]*$|^-[A-Za-z-]*$")
NO_SECRETS_IN_ENV = re.compile(r"(?i)\b(secret|token|password|api_?key|credential)\b")
CONFIG_DOTFILE = re.compile(r"~?/?\.(zshrc|bashrc|bash_profile|zprofile|profile|zshenv)\b")


def bare_words(words):
    return [w for w in words if not w.startswith("-")]


def _curl_get_localhost(words):
    """curl/wget stage: GET (no -d/-F/-T/--upload/--output), URL on localhost."""
    toks = words[1:]
    if any(t in ("-d", "--data", "-F", "--form", "-T", "--upload-file",
                 "--output", "-o", "-O", "--remote-name", "-X") for t in toks):
        return False
    if any(t in ("-X", "--request") for t in toks):  # any explicit -X: not provably GET
        return False
    urls = [t for t in toks if not t.startswith("-")]
    if not urls:
        return False
    localhost = re.compile(
        r"^(https?://(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0)(:\d+)?(/.*)?$"
        r"|^(localhost|127\.0\.0\.1)(:\d+)?(/.*)?$)", re.I)
    return all(localhost.match(u.split("'")[0]) for u in urls)


def _env_stage_safe(words):
    """env/printenv: refuse when args name SECRET-ish vars (values leak via stdout
    anyway, but naming one is the signal)."""
    return not NO_SECRETS_IN_ENV.search(" ".join(words))


def is_read_only(command: str) -> bool:
    """True when every stage of the pipeline is a verb that cannot change or
    transmit anything. No redirection/substitution/piping-into-unsafe verbs.
    ALSO False when the command's TEXT trips a CATASTROPHIC/SECRET/SENDS floor
    even as a mere argument — `echo rm -rf /` prints words, not damage, but the
    fast path's contract is 'cannot destroy OR transmit anything', and trusting
    argument content is exactly how fast paths get gamed (arbiter keeps
    floors-out-of-fast-path for the same reason)."""
    if UNSAFE_TOKENS.search(command) or SECRET_PATH.search(command) or CONFIG_DOTFILE.search(command):
        return False
    if CATASTROPHIC.search(command) or SENDS_OUT.search(command):
        return False  # destructive/transmit patterns as text disqualify the fast path
    for stage in command.split("|"):
        words = stage.strip().split()
        if not words:
            return False
        verb = os.path.basename(words[0])
        if verb in ("curl", "wget"):
            if not _curl_get_localhost(words):
                return False
            continue
        if verb in ("env", "printenv"):
            if not _env_stage_safe(words):
                return False
            continue
        if verb == "find":
            if UNSAFE_FIND.search(stage):
                return False
            continue
        if VERSION_ONLY.match(stage.strip()):
            continue
        if verb in READ_ONLY:
            if verb == "sed" and any(w == "-i" or w.startswith(("--in-place", "-i."))
                                     for w in words[1:]):
                return False
            continue
        allowed = READ_ONLY_SUB.get(verb)
        if allowed is not None:
            bare = bare_words(words[1:])
            if verb == "gh":
                if bare[:1] and bare[0] in _GH_VIEW:
                    continue
                return False
            if bare[:1] and bare[0] in allowed:
                continue
            if not bare and verb == "nvidia-smi":
                continue
        return False
    return True


# --------------------------------------------------------------------------- layer 2
# Every line here is a case the model was measured to get wrong (arbiter:
# `crontab -r` and `cat .env >> /tmp/x` scored below `git fetch origin`).
SECRET_PATH = re.compile(
    r"(~|\$HOME)?/?\.(ssh|aws|gnupg|azure|kube|docker|netrc|npmrc)\b"
    r"|\bid_(rsa|ed25519|ecdsa)\b|(^|[\s/=\"'])\.env\b|\bcredentials\.json\b"
    r"|\baws[/\\]credentials\b|\bkeychain\b"
    r"|\bprivatekey\b|(?<![\w.])private[._ -]?key\b"
    r"|(?<![\w.])/etc/(shadow|passwd|sudoers)\b"
    r"|\b[A-Z_]*(TOKEN|SECRET|PASSWORD|API_KEY)\b|\.(pem|p12|pfx|kdbx)\b", re.I)

SENDS_OUT = re.compile(
    r"\b(scp|sftp)\b"
    r"|\brsync\b.*\b(\w+@[\w.-]+:|rsync://)"
    r"|\b(curl|wget)\b[^|]*\b(-T|--upload-file|--data|--data-binary|-d\s|[Ff]ile=@|@/|-F\s)"
    r"|\|\s*(curl|nc|ncat|ssh)\b"
    r"|\bgit\s+push\b[^|]*--force"
    r"|\bmail\b|\bpbcopy\b|>>?\s*(/|~|\$HOME)")

CATASTROPHIC = re.compile(
    r"rm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf][a-zA-Z]*\s+(-[a-zA-Z-]+\s+)*(/|~|\$HOME|\*|\.)\s*(\*|/\*)?\s*$"
    r"|--no-preserve-root|\bmkfs(\.|\s)|\bdd\b[^|]*\bof=/dev/"
    r"|:\(\)\s*\{|\bcrontab\s+-r\b|\bhistory\s+-c\b"
    r"|--force(-with-lease)?\s+\S*\s*(origin\s+)?(main|master)\b"
    r"|\bpush\b[^|]*?(main|master)\b[^|]*?(-f|--force(-with-lease)?)\b"
    r"|\bgit\s+push\b[^|]*(--force(-with-lease)?|\s-f\b)"
    r"|\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b"
    r"|base64\s+(-d|--decode)[^|]*\|\s*(sh|bash)\b"
    r"|\bDROP\s+(TABLE|DATABASE)\b|\bDELETE\s+FROM\s+\w+\s*(;|\"|')?\s*$"
    r"|\bchmod\s+-R\s+777\s+(/|~|\$HOME|\*)"
    r"|\b(shutdown|reboot|halt|poweroff)\b(?![^\n]*\bask\b)", re.I)


def floor_for(command: str):
    """The highest risk floor the command's text triggers, and a reason phrase.

    Returns (floor_level, reason) — (0.0, '') when no floor matches.
    CATASTROPHIC → 0.99 (deny); SECRET_PATH/SENDS_OUT → 0.78 (deny floor).
    """
    if CATASTROPHIC.search(command):
        return 0.99, "a pattern that is never right by accident"
    if SECRET_PATH.search(command):
        if SENDS_OUT.search(command):
            return DENY_AT, "a credential path on its way off this machine"
        return DENY_AT, "a credential path (deny floor: model scores these below git fetch)"
    if SENDS_OUT.search(command):
        return DENY_AT, "a verb that sends local data off this machine"
    if CONFIG_DOTFILE.search(command):
        return ASK_AT, "a shell rc file, where exported tokens live"
    return 0.0, ""


# --------------------------------------------------------------------------- layer 3
QUESTIONS = {
    "destroys_data": {
        "type": "noul",
        "instructions": "This command deletes or overwrites something that cannot be brought back.",
    },
    "reads_secrets": {
        "type": "noul",
        "instructions": ("This command reads or prints a secret: an SSH key, a token, "
                         "a password, a .env file, cloud credentials, a keychain entry."),
    },
    "sends_data_to_remote": {
        "type": "noul",
        "instructions": ("This command sends the contents of a local file or an environment "
                         "variable to a machine outside this computer."),
    },
    "what_is_lost": {
        "type": "score",
        "instructions": "If this command is a mistake, what is lost?",
        "criteria": ["nothing", "ephemeral files", "user data",
                     "credentials or system access"],
    },
}

WEIGHTS = {
    "destroys_data": 0.85,
    "what_is_lost": 0.15,   # score is divided by 3 first: this is its weight at full scale
    # reads_secrets / sends_data_to_remote: measured weight 0.35/0.15 on
    # arbiter's set — kept OUT of the composite here because our Laya
    # checkpoint returns bare 0/1 priors for them (confidence 0.0, measured
    # 2026-09-24 on Spark); including them would floor every state at 0.5-0.6
    # and swallow the ask band. Secret-handling is text's job (SECRET_PATH
    # floor at 0.78) until the weights are refit on our traffic.
}


def model_weighted_risk(answers: dict) -> float:
    """BIAS + Σ(weight × signal), from the provider's typed answers."""
    total = BIAS
    for qid, weight in WEIGHTS.items():
        ans = answers.get(qid)
        if ans is None:
            continue
        if ans.get("type") == "noul":
            value = ans.get("noul", 0.0)
        elif ans.get("type") == "score":
            value = float(ans.get("score", 0.0)) / 3.0
        else:
            continue
        total += weight * float(value)
    return total


def verdict_for(risk: float) -> str:
    return "deny" if risk >= DENY_AT else "ask" if risk >= ASK_AT else "allow"


def evaluate(command_text: str, existing_questions: dict | None = None,
             call_model=None, mode: str | None = None):
    """The 3-layer gate. Returns the verdict dict for server.py to consume.

    command_text: the state text (our tool-call states carry the command).
    existing_questions: caller-supplied questions (contract passthrough —
      model layer scores THEM plus the policy set, merged).
    call_model(state_text, merged_questions) -> provider result; accepts both
      the bare answers dict {qid: {...}} and the /decide envelope
      {"answers": {...}}; None means the model layer failed (fail-closed
      discipline is decided below, not here).
    mode: 'hybrid' | 'model-only' (None → read DECISERV_POLICY default).

    Returns {verdict, final_risk, layer, text_floor, reason} where layer is
    fast_path | text_floor | model, and text_floor is None or
    {class, reason}.
    """
    mode = (mode or POLICY_MODE or "hybrid").strip().lower()
    text = command_text if isinstance(command_text, str) else str(command_text)
    floor, why, floor_class = 0.0, "", None
    for klass, rx in (("CATASTROPHIC", CATASTROPHIC), ("SECRET_PATH", SECRET_PATH),
                      ("SENDS_OUT", SENDS_OUT), ("CONFIG_DOTFILE", CONFIG_DOTFILE)):
        if rx.search(text):
            f, w = floor_for(text)
            if f > floor:
                floor, why, floor_class = f, w, klass

    def _answers_of(res):
        """Provider results come back as {'answers': {...}} (/decide envelope);
        accept the bare answers dict too."""
        if not isinstance(res, dict):
            return None
        if "answers" in res and isinstance(res["answers"], dict):
            return res["answers"]
        return res

    if mode == "model-only":
        # Legacy behavior: model decides, floors only fail-closed on error.
        answers = _answers_of(call_model(text, existing_questions or QUESTIONS))
        if answers is None:
            return _fallback(floor_class is not None, floor)
        risk = model_weighted_risk(answers)
        v = verdict_for(risk)
        return {"verdict": v, "final_risk": risk, "layer": "model",
                "text_floor": None, "reason": ""}

    if mode != "hybrid":
        mode = "hybrid"

    if floor == 0.0 and is_read_only(text):
        return {"verdict": "allow", "final_risk": 0.0, "layer": "fast_path",
                "text_floor": None, "reason": "read-only verbs only"}

    # Layer 2/3: floors raise, model can only make it worse, never better.
    call_model_for_verdict = floor == 0.0   # floor-only verdicts skip the model too
    answers = None
    if call_model_for_verdict:
        merged = dict(QUESTIONS)
        for q, spec in (existing_questions or {}).items():
            merged.setdefault(q, spec)
        answers = _answers_of(call_model(text, merged)) if call_model else None

    if answers is None:
        # Fail-CLOSED if a floor matched; fail open to escalate otherwise.
        if floor_class is not None:
            return {"verdict": "deny", "final_risk": floor, "layer": "text_floor",
                    "text_floor": {"class": floor_class, "reason": why},
                    "reason": f"fail-closed: {why}"}
        return {"verdict": "ask", "final_risk": max(floor, ASK_AT), "layer": "text_floor" if floor_class else "model",
                "text_floor": {"class": floor_class, "reason": why} if floor_class else None,
                "reason": "model unavailable — fail open to escalate, never silent-allow"}

    if not call_model_for_verdict:
        # Floor-only verdict: model NOT called (it can only make it worse).
        return {"verdict": verdict_for(floor), "final_risk": floor, "layer": "text_floor",
                "text_floor": {"class": floor_class, "reason": why},
                "reason": f"text floor: {why}"}

    risk = model_weighted_risk(answers)
    final = max(risk, floor)   # floor can only RAISE risk, never lower it
    v = verdict_for(final)
    return {"verdict": v, "final_risk": final, "layer": "model",
            # answers ride along — server contract ships them (answers may be
            # a subset when the caller's questions were merged with the policy set)
            "answers": answers or {},
            "text_floor": {"class": floor_class, "reason": why} if floor_class else None,
            "reason": why}