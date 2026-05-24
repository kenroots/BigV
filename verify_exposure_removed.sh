#!/usr/bin/env bash
# =============================================================================
# verify_exposure_removed.sh
# =============================================================================
# Purpose : Confirm that the exposed IBM Cloud IAM API key is no longer
#           accessible / active and that the offending commit content has
#           been addressed.
#
# Exposure details
#   Token Owner  : cheruiyo@us.ibm.com
#   Repo         : cheruiyo/gaudi3  (github.ibm.com)
#   File         : setup_env_vars.sh  (line 31)
#   Commit       : 2dbed40beca260b304917c0f035cd55c871dcf5f
#   Date found   : 2026-02-10T00:36:01+00:00
#   Key ID       : ApiKey-1ce6f235-41b8-4175-9397-46a987c7e378
#   Key Name     : HPC-Debug-Agent
#   IAM ID       : IBMid-50XPUV907C
#   Account ID   : f01e9db09581e4124215e74f99c9a247
# =============================================================================

set -euo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass()  { echo -e "${GREEN}[PASS]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; FAILURES=$((FAILURES+1)); }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
header(){ echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"; \
          echo -e "${BOLD}${CYAN}  $*${RESET}"; \
          echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"; }

FAILURES=0

# ── configuration ─────────────────────────────────────────────────────────────
GITHUB_HOST="github.ibm.com"
REPO="cheruiyo/gaudi3"
COMMIT="2dbed40beca260b304917c0f035cd55c871dcf5f"
EXPOSED_FILE="setup_env_vars.sh"
EXPOSED_LINE=31
KEY_ID="ApiKey-1ce6f235-41b8-4175-9397-46a987c7e378"
KEY_CRN="crn:v1:bluemix:public:iam-identity::a/f01e9db09581e4124215e74f99c9a247F::apikey:${KEY_ID}"
KEY_NAME="HPC-Debug-Agent"
ACCOUNT_ID="f01e9db09581e4124215e74f99c9a247"

# Optional: set these env vars before running, or the script will prompt.
#   GH_IBM_TOKEN   – GitHub Enterprise personal access token (read:repo)
#   IBMCLOUD_API_KEY – An *admin* IBM Cloud API key for the account above
#                      (NOT the exposed key – a separate key with IAM read access)

# ── prerequisite checks ───────────────────────────────────────────────────────
header "0. Prerequisite checks"

for cmd in curl jq git; do
  if command -v "$cmd" &>/dev/null; then
    pass "$cmd is available ($(command -v $cmd))"
  else
    fail "$cmd is NOT installed – please install it and re-run."
  fi
done

if command -v ibmcloud &>/dev/null; then
  pass "ibmcloud CLI is available"
  IBMCLOUD_AVAILABLE=true
else
  warn "ibmcloud CLI not found – IBM Cloud checks will be skipped."
  warn "Install: https://cloud.ibm.com/docs/cli"
  IBMCLOUD_AVAILABLE=false
fi

# ── token prompts (if not already set) ───────────────────────────────────────
if [[ -z "${GH_IBM_TOKEN:-}" ]]; then
  echo ""
  read -rsp "Enter your GitHub Enterprise (${GITHUB_HOST}) personal access token: " GH_IBM_TOKEN
  echo ""
fi

if [[ "${IBMCLOUD_AVAILABLE}" == "true" && -z "${IBMCLOUD_API_KEY:-}" ]]; then
  echo ""
  warn "IBMCLOUD_API_KEY not set. IBM Cloud checks require an admin/viewer key"
  warn "(NOT the exposed key). Leave blank to skip IBM Cloud checks."
  read -rsp "Enter IBM Cloud admin API key (or press Enter to skip): " IBMCLOUD_API_KEY
  echo ""
fi

# =============================================================================
# CHECK 1 – GitHub: does the file still contain the key on the default branch?
# =============================================================================
header "1. GitHub – current HEAD of default branch"

info "Fetching default branch info for ${REPO} …"
REPO_INFO=$(curl -sf \
  -H "Authorization: token ${GH_IBM_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://${GITHUB_HOST}/api/v3/repos/${REPO}" 2>/dev/null || echo "{}")

DEFAULT_BRANCH=$(echo "$REPO_INFO" | jq -r '.default_branch // "main"')
info "Default branch: ${DEFAULT_BRANCH}"

info "Fetching raw content of ${EXPOSED_FILE} from HEAD …"
RAW_CONTENT=$(curl -sf \
  -H "Authorization: token ${GH_IBM_TOKEN}" \
  -H "Accept: application/vnd.github.v3.raw" \
  "https://${GITHUB_HOST}/api/v3/repos/${REPO}/contents/${EXPOSED_FILE}?ref=${DEFAULT_BRANCH}" \
  2>/dev/null || echo "FILE_NOT_FOUND")

if [[ "$RAW_CONTENT" == "FILE_NOT_FOUND" ]]; then
  pass "${EXPOSED_FILE} no longer exists on branch '${DEFAULT_BRANCH}' (or access denied)."
else
  # Look for patterns that look like IBM Cloud API keys (44-char base64-ish strings)
  # IBM Cloud API keys typically match: [A-Za-z0-9_\-]{44}
  APIKEY_PATTERN='[A-Za-z0-9_\-]{44}'
  FOUND_KEYS=$(echo "$RAW_CONTENT" | grep -oE "$APIKEY_PATTERN" | sort -u || true)

  if [[ -z "$FOUND_KEYS" ]]; then
    pass "No IBM Cloud API key pattern found in current HEAD of ${EXPOSED_FILE}."
  else
    fail "Potential API key(s) still present in ${EXPOSED_FILE} on branch '${DEFAULT_BRANCH}':"
    echo "$FOUND_KEYS" | while read -r k; do
      echo "       → ${k:0:8}…${k: -4} (redacted)"
    done
  fi

  # Check line 31 specifically
  LINE31=$(echo "$RAW_CONTENT" | sed -n "${EXPOSED_LINE}p")
  if echo "$LINE31" | grep -qE "$APIKEY_PATTERN"; then
    fail "Line ${EXPOSED_LINE} still contains an API key pattern: $(echo "$LINE31" | sed 's/[A-Za-z0-9_\-]\{44\}/[REDACTED]/g')"
  else
    pass "Line ${EXPOSED_LINE} does not contain an API key pattern."
  fi
fi

# =============================================================================
# CHECK 2 – GitHub: inspect the offending commit directly
# =============================================================================
header "2. GitHub – offending commit content"

info "Fetching diff for commit ${COMMIT:0:12}… …"
COMMIT_DATA=$(curl -sf \
  -H "Authorization: token ${GH_IBM_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://${GITHUB_HOST}/api/v3/repos/${REPO}/commits/${COMMIT}" \
  2>/dev/null || echo "{}")

COMMIT_MSG=$(echo "$COMMIT_DATA" | jq -r '.commit.message // "unknown"')
info "Commit message: ${COMMIT_MSG}"

# Check if a revert / remediation commit exists after the offending one
info "Checking for remediation commits after ${COMMIT:0:12}… …"
COMMITS_AFTER=$(curl -sf \
  -H "Authorization: token ${GH_IBM_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://${GITHUB_HOST}/api/v3/repos/${REPO}/commits?path=${EXPOSED_FILE}&per_page=10" \
  2>/dev/null || echo "[]")

COMMIT_COUNT=$(echo "$COMMITS_AFTER" | jq 'length' 2>/dev/null || echo 0)
info "Found ${COMMIT_COUNT} commit(s) touching ${EXPOSED_FILE}."

LATEST_COMMIT=$(echo "$COMMITS_AFTER" | jq -r '.[0].sha // "unknown"')
if [[ "$LATEST_COMMIT" != "$COMMIT" && "$LATEST_COMMIT" != "unknown" ]]; then
  pass "Newer commits exist after the offending commit (latest: ${LATEST_COMMIT:0:12}…)."
  info "Review them at: https://${GITHUB_HOST}/${REPO}/commits/${DEFAULT_BRANCH}/${EXPOSED_FILE}"
else
  warn "The offending commit appears to be the most recent change to ${EXPOSED_FILE}."
  warn "Verify manually that the key has been removed or the file deleted."
fi

# =============================================================================
# CHECK 3 – GitHub: confirm history rewrite (force-push / BFG)
# =============================================================================
header "3. GitHub – history rewrite verification"

info "Checking if the offending commit hash still resolves …"
COMMIT_CHECK=$(curl -sf \
  -H "Authorization: token ${GH_IBM_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://${GITHUB_HOST}/api/v3/repos/${REPO}/git/commits/${COMMIT}" \
  2>/dev/null || echo "{}")

COMMIT_SHA=$(echo "$COMMIT_CHECK" | jq -r '.sha // "not_found"')
if [[ "$COMMIT_SHA" == "not_found" || "$COMMIT_SHA" == "null" ]]; then
  pass "Commit ${COMMIT:0:12}… no longer exists in the repository – history has been rewritten."
else
  warn "Commit ${COMMIT:0:12}… still exists in the repository."
  warn "The secret is still in git history. Consider:"
  warn "  • BFG Repo Cleaner  : https://rtyley.github.io/bfg-repo-cleaner/"
  warn "  • git filter-repo   : https://github.com/newren/git-filter-repo"
  warn "  • GitHub support to purge cached views"
fi

# =============================================================================
# CHECK 4 – IBM Cloud: verify the API key is revoked/deleted
# =============================================================================
header "4. IBM Cloud – API key status"

if [[ "${IBMCLOUD_AVAILABLE}" == "false" || -z "${IBMCLOUD_API_KEY:-}" ]]; then
  warn "Skipping IBM Cloud checks (ibmcloud CLI unavailable or no admin key provided)."
else
  info "Logging in to IBM Cloud …"
  if ibmcloud login --apikey "${IBMCLOUD_API_KEY}" -a https://cloud.ibm.com \
       --no-region -q 2>/dev/null; then
    pass "IBM Cloud login successful."

    info "Querying API key ${KEY_ID} …"
    KEY_STATUS=$(ibmcloud iam api-key-get "${KEY_NAME}" --output json 2>/dev/null || echo "{}")
    KEY_STATE=$(echo "$KEY_STATUS" | jq -r '.state // "not_found"')

    case "$KEY_STATE" in
      "not_found"|"")
        pass "API key '${KEY_NAME}' (${KEY_ID}) not found – it has been deleted. ✓"
        ;;
      "active")
        fail "API key '${KEY_NAME}' is still ACTIVE. Revoke it immediately:"
        fail "  ibmcloud iam api-key-delete '${KEY_NAME}' -f"
        ;;
      "inactive"|"locked")
        warn "API key '${KEY_NAME}' exists but is ${KEY_STATE}. Recommend full deletion:"
        warn "  ibmcloud iam api-key-delete '${KEY_NAME}' -f"
        ;;
      *)
        warn "API key state: ${KEY_STATE} – verify manually."
        ;;
    esac

    # Also try direct CRN lookup via IAM API
    info "Cross-checking via IBM Cloud IAM REST API …"
    IAM_TOKEN=$(ibmcloud iam oauth-tokens --output json 2>/dev/null \
                | jq -r '.iam_token // ""')

    if [[ -n "$IAM_TOKEN" ]]; then
      KEY_DETAIL=$(curl -sf \
        -H "Authorization: ${IAM_TOKEN}" \
        -H "Content-Type: application/json" \
        "https://iam.cloud.ibm.com/v1/apikeys/${KEY_ID}" 2>/dev/null || echo "{}")

      REST_STATE=$(echo "$KEY_DETAIL" | jq -r '.state // "not_found"')
      if [[ "$REST_STATE" == "not_found" || -z "$REST_STATE" ]]; then
        pass "IAM REST API: key ${KEY_ID} not found – confirmed deleted. ✓"
      else
        fail "IAM REST API: key ${KEY_ID} state = ${REST_STATE}"
      fi
    fi

    ibmcloud logout -q 2>/dev/null || true
  else
    fail "IBM Cloud login failed – check your admin API key."
  fi
fi

# =============================================================================
# CHECK 5 – Local git clone verification (optional)
# =============================================================================
header "5. Local git clone – grep for key patterns"

CLONE_DIR=$(mktemp -d)
info "Cloning ${REPO} into ${CLONE_DIR} …"

if git clone --quiet \
     "https://x-token-auth:${GH_IBM_TOKEN}@${GITHUB_HOST}/${REPO}.git" \
     "${CLONE_DIR}" 2>/dev/null; then
  pass "Clone successful."

  info "Scanning working tree for IBM Cloud API key patterns …"
  GREP_HITS=$(grep -rn --include="*.sh" --include="*.env" --include="*.yaml" \
    --include="*.yml" --include="*.json" --include="*.tf" \
    -E '[A-Za-z0-9_\-]{44}' "${CLONE_DIR}" 2>/dev/null \
    | grep -v ".git/" || true)

  if [[ -z "$GREP_HITS" ]]; then
    pass "No API key patterns found in working tree."
  else
    fail "Potential API key patterns found in working tree:"
    echo "$GREP_HITS" | sed 's/[A-Za-z0-9_\-]\{44\}/[REDACTED]/g' | head -20
  fi

  info "Scanning full git history for key patterns (this may take a moment) …"
  HISTORY_HITS=$(git -C "${CLONE_DIR}" log --all -p -- "${EXPOSED_FILE}" 2>/dev/null \
    | grep -E '^\+.*[A-Za-z0-9_\-]{44}' | grep -v "^+++" || true)

  if [[ -z "$HISTORY_HITS" ]]; then
    pass "No API key patterns found in git history for ${EXPOSED_FILE}."
  else
    warn "API key patterns still present in git history for ${EXPOSED_FILE}:"
    echo "$HISTORY_HITS" | sed 's/[A-Za-z0-9_\-]\{44\}/[REDACTED]/g' | head -10
    warn "History rewrite (BFG / filter-repo) is required to fully purge the secret."
  fi

  rm -rf "${CLONE_DIR}"
else
  warn "Could not clone repository – skipping local scan."
  rm -rf "${CLONE_DIR}"
fi

# =============================================================================
# SUMMARY
# =============================================================================
header "Summary"

echo ""
echo -e "  Repo    : https://${GITHUB_HOST}/${REPO}"
echo -e "  Commit  : ${COMMIT}"
echo -e "  File    : ${EXPOSED_FILE}  (line ${EXPOSED_LINE})"
echo -e "  Key ID  : ${KEY_ID}"
echo -e "  Key Name: ${KEY_NAME}"
echo ""

if [[ $FAILURES -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}  ✅  All checks passed – exposure appears to be remediated.${RESET}"
else
  echo -e "${RED}${BOLD}  ❌  ${FAILURES} check(s) FAILED – remediation is INCOMPLETE.${RESET}"
  echo ""
  echo -e "${YELLOW}  Recommended remediation steps:${RESET}"
  echo "  1. Delete the exposed API key immediately:"
  echo "     ibmcloud iam api-key-delete '${KEY_NAME}' -f"
  echo ""
  echo "  2. Remove the secret from the file and push a fix commit."
  echo ""
  echo "  3. Rewrite git history to purge the secret from all commits:"
  echo "     # Using BFG Repo Cleaner:"
  echo "     echo 'EXPOSED_KEY_VALUE' > secrets.txt"
  echo "     bfg --replace-text secrets.txt ${REPO##*/}.git"
  echo "     git reflog expire --expire=now --all"
  echo "     git gc --prune=now --aggressive"
  echo "     git push --force --all"
  echo ""
  echo "  4. Contact GitHub Enterprise admins to purge cached commit views."
  echo ""
  echo "  5. Rotate any services/resources that used the HPC-Debug-Agent key."
  echo ""
  echo "  6. Audit IBM Cloud activity logs for unauthorized usage:"
  echo "     ibmcloud at events --start \$(date -d '2026-02-06' +%Y-%m-%d) \\"
  echo "                        --end   \$(date +%Y-%m-%d)"
fi

echo ""
exit $FAILURES

# Made with Bob
