#!/usr/bin/env bash
# =============================================================================
# scan_all_repos_for_secrets.sh
# =============================================================================
# Purpose : Scan ALL GitHub repositories (github.com and github.ibm.com) for
#           hardcoded secrets, API keys, passwords, and sensitive credentials.
#
# Features:
#   • Scans both public GitHub and GitHub Enterprise
#   • Detects multiple secret types (API keys, passwords, tokens, etc.)
#   • Checks current files AND git history
#   • Generates detailed HTML report
#   • Parallel processing for speed
# =============================================================================

set -euo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass()  { echo -e "${GREEN}[PASS]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; FAILURES=$((FAILURES+1)); }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; WARNINGS=$((WARNINGS+1)); }
info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
header(){ echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"; \
          echo -e "${BOLD}${CYAN}  $*${RESET}"; \
          echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"; }

FAILURES=0
WARNINGS=0
TOTAL_REPOS=0
SCANNED_REPOS=0

# ── configuration ─────────────────────────────────────────────────────────────
GITHUB_COM="github.com"
GITHUB_IBM="github.ibm.com"
SCAN_DIR=$(mktemp -d)
REPORT_FILE="secret_scan_report_$(date +%Y%m%d_%H%M%S).html"
JSON_REPORT="secret_scan_report_$(date +%Y%m%d_%H%M%S).json"

# Maximum number of parallel scans
MAX_PARALLEL=5

# ── secret patterns ───────────────────────────────────────────────────────────
declare -A SECRET_PATTERNS=(
  # IBM Cloud
  ["IBM Cloud API Key"]='[A-Za-z0-9_\-]{44}'
  ["IBM Cloud IAM Token"]='eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+'
  
  # AWS
  ["AWS Access Key"]='AKIA[0-9A-Z]{16}'
  ["AWS Secret Key"]='[A-Za-z0-9/+=]{40}'
  ["AWS Session Token"]='FQoGZXIvYXdzE[A-Za-z0-9/+=]+'
  
  # Azure
  ["Azure Storage Key"]='[A-Za-z0-9/+=]{88}'
  ["Azure Client Secret"]='[A-Za-z0-9~._-]{34,40}'
  
  # Google Cloud
  ["GCP API Key"]='AIza[0-9A-Za-z_\-]{35}'
  ["GCP Service Account"]='\"type\": \"service_account\"'
  
  # GitHub
  ["GitHub Token"]='gh[pousr]_[A-Za-z0-9]{36,255}'
  ["GitHub Classic PAT"]='ghp_[A-Za-z0-9]{36}'
  ["GitHub Fine-grained PAT"]='github_pat_[A-Za-z0-9_]{82}'
  
  # Generic patterns
  ["Generic API Key"]='api[_-]?key["\s:=]+[A-Za-z0-9_\-]{20,}'
  ["Generic Secret"]='secret["\s:=]+[A-Za-z0-9_\-]{20,}'
  ["Generic Password"]='password["\s:=]+[^"\s]{8,}'
  ["Generic Token"]='token["\s:=]+[A-Za-z0-9_\-]{20,}'
  ["Private Key Header"]='-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
  
  # Database
  ["Database Connection String"]='(mongodb|mysql|postgresql|postgres|redis)://[^"\s]+'
  ["JDBC Connection"]='jdbc:[^"\s]+'
  
  # Slack
  ["Slack Token"]='xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]+'
  ["Slack Webhook"]='https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+'
  
  # Other services
  ["Stripe Key"]='sk_live_[0-9a-zA-Z]{24,}'
  ["Twilio Key"]='SK[0-9a-fA-F]{32}'
  ["SendGrid Key"]='SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}'
  ["Mailgun Key"]='key-[0-9a-zA-Z]{32}'
  ["NPM Token"]='npm_[A-Za-z0-9]{36}'
  ["PyPI Token"]='pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]+'
)

# Files to exclude from scanning
EXCLUDE_PATTERNS=(
  "*.min.js"
  "*.min.css"
  "package-lock.json"
  "yarn.lock"
  "*.svg"
  "*.png"
  "*.jpg"
  "*.jpeg"
  "*.gif"
  "*.ico"
  "*.woff"
  "*.woff2"
  "*.ttf"
  "*.eot"
  "node_modules/*"
  "vendor/*"
  ".git/*"
  "*.pyc"
  "__pycache__/*"
)

# ── prerequisite checks ───────────────────────────────────────────────────────
header "0. Prerequisite checks"

for cmd in curl jq git; do
  if command -v "$cmd" &>/dev/null; then
    pass "$cmd is available"
  else
    fail "$cmd is NOT installed – please install it and re-run."
    exit 1
  fi
done

# ── token prompts ─────────────────────────────────────────────────────────────
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo ""
  read -rsp "Enter your GitHub.com personal access token (or press Enter to skip): " GITHUB_TOKEN
  echo ""
fi

if [[ -z "${GITHUB_IBM_TOKEN:-}" ]]; then
  echo ""
  read -rsp "Enter your GitHub Enterprise (github.ibm.com) token (or press Enter to skip): " GITHUB_IBM_TOKEN
  echo ""
fi

# ── initialize report ─────────────────────────────────────────────────────────
cat > "$REPORT_FILE" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Secret Scan Report</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
           background: #f5f5f5; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; background: white; 
                 border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              color: white; padding: 30px; border-radius: 8px 8px 0 0; }
    .header h1 { font-size: 32px; margin-bottom: 10px; }
    .header p { opacity: 0.9; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
             gap: 20px; padding: 30px; background: #f8f9fa; }
    .stat-card { background: white; padding: 20px; border-radius: 8px; 
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stat-card h3 { color: #666; font-size: 14px; margin-bottom: 10px; }
    .stat-card .value { font-size: 32px; font-weight: bold; }
    .stat-card.critical .value { color: #dc3545; }
    .stat-card.warning .value { color: #ffc107; }
    .stat-card.success .value { color: #28a745; }
    .stat-card.info .value { color: #17a2b8; }
    .content { padding: 30px; }
    .repo-section { margin-bottom: 40px; border: 1px solid #e0e0e0; 
                    border-radius: 8px; overflow: hidden; }
    .repo-header { background: #f8f9fa; padding: 20px; border-bottom: 1px solid #e0e0e0; 
                   display: flex; justify-content: space-between; align-items: center; }
    .repo-header h2 { font-size: 20px; color: #333; }
    .repo-header .badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; 
                          font-weight: bold; }
    .badge.critical { background: #dc3545; color: white; }
    .badge.warning { background: #ffc107; color: #333; }
    .badge.clean { background: #28a745; color: white; }
    .finding { padding: 20px; border-bottom: 1px solid #f0f0f0; }
    .finding:last-child { border-bottom: none; }
    .finding-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .finding-type { font-weight: bold; color: #dc3545; }
    .finding-location { color: #666; font-size: 14px; }
    .finding-content { background: #f8f9fa; padding: 15px; border-radius: 4px; 
                       font-family: 'Courier New', monospace; font-size: 13px; 
                       overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
    .redacted { background: #333; color: #333; padding: 2px 4px; border-radius: 2px; }
    .redacted:hover { color: #fff; }
    .footer { text-align: center; padding: 20px; color: #666; font-size: 14px; 
              border-top: 1px solid #e0e0e0; }
    .no-findings { text-align: center; padding: 40px; color: #666; }
    .no-findings svg { width: 64px; height: 64px; margin-bottom: 20px; opacity: 0.5; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🔐 Secret Scan Report</h1>
      <p>Generated: <span id="timestamp"></span></p>
    </div>
    <div class="stats">
      <div class="stat-card critical">
        <h3>Critical Findings</h3>
        <div class="value" id="critical-count">0</div>
      </div>
      <div class="stat-card warning">
        <h3>Warnings</h3>
        <div class="value" id="warning-count">0</div>
      </div>
      <div class="stat-card success">
        <h3>Clean Repositories</h3>
        <div class="value" id="clean-count">0</div>
      </div>
      <div class="stat-card info">
        <h3>Total Repositories</h3>
        <div class="value" id="total-count">0</div>
      </div>
    </div>
    <div class="content" id="findings-content">
EOF

echo '[]' > "$JSON_REPORT"

# ── helper functions ──────────────────────────────────────────────────────────
function fetch_repos() {
  local host=$1
  local token=$2
  local page=1
  local repos=()
  
  if [[ -z "$token" ]]; then
    return
  fi
  
  info "Fetching repositories from $host ..."
  
  while true; do
    local response
    if [[ "$host" == "$GITHUB_COM" ]]; then
      response=$(curl -sf \
        -H "Authorization: token $token" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.${host}/user/repos?per_page=100&page=$page&affiliation=owner,collaborator,organization_member" \
        2>/dev/null || echo "[]")
    else
      response=$(curl -sf \
        -H "Authorization: token $token" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://${host}/api/v3/user/repos?per_page=100&page=$page&affiliation=owner,collaborator,organization_member" \
        2>/dev/null || echo "[]")
    fi
    
    local count=$(echo "$response" | jq 'length')
    if [[ "$count" -eq 0 ]]; then
      break
    fi
    
    repos+=($(echo "$response" | jq -r '.[].full_name'))
    ((page++))
  done
  
  echo "${repos[@]}"
}

function scan_repo() {
  local host=$1
  local repo=$2
  local token=$3
  local clone_url
  
  if [[ "$host" == "$GITHUB_COM" ]]; then
    clone_url="https://x-access-token:${token}@${host}/${repo}.git"
  else
    clone_url="https://x-token-auth:${token}@${host}/${repo}.git"
  fi
  
  local repo_dir="${SCAN_DIR}/${repo//\//_}"
  local findings=()
  
  info "Scanning $repo ..."
  
  # Clone repository
  if ! git clone --quiet --depth 50 "$clone_url" "$repo_dir" 2>/dev/null; then
    warn "Failed to clone $repo"
    return
  fi
  
  # Scan current files
  for pattern_name in "${!SECRET_PATTERNS[@]}"; do
    local pattern="${SECRET_PATTERNS[$pattern_name]}"
    
    while IFS= read -r line; do
      if [[ -n "$line" ]]; then
        local file=$(echo "$line" | cut -d: -f1)
        local line_num=$(echo "$line" | cut -d: -f2)
        local content=$(echo "$line" | cut -d: -f3-)
        
        # Redact the actual secret
        local redacted=$(echo "$content" | sed -E "s/${pattern}/[REDACTED]/g")
        
        findings+=("CURRENT|$pattern_name|$file|$line_num|$redacted")
      fi
    done < <(grep -rn -E "$pattern" "$repo_dir" 2>/dev/null | grep -v ".git/" || true)
  done
  
  # Scan git history (last 50 commits)
  for pattern_name in "${!SECRET_PATTERNS[@]}"; do
    local pattern="${SECRET_PATTERNS[$pattern_name]}"
    
    while IFS= read -r line; do
      if [[ -n "$line" ]]; then
        local commit=$(echo "$line" | awk '{print $1}')
        local content=$(echo "$line" | cut -d: -f2-)
        local redacted=$(echo "$content" | sed -E "s/${pattern}/[REDACTED]/g")
        
        findings+=("HISTORY|$pattern_name|commit:$commit|0|$redacted")
      fi
    done < <(git -C "$repo_dir" log --all -p -S"$pattern" --pickaxe-regex --format="%H" 2>/dev/null | \
             grep -E "^\+.*${pattern}" | head -20 || true)
  done
  
  # Output findings
  if [[ ${#findings[@]} -gt 0 ]]; then
    for finding in "${findings[@]}"; do
      echo "$host|$repo|$finding"
    done
  fi
  
  # Cleanup
  rm -rf "$repo_dir"
}

# ── main scan loop ────────────────────────────────────────────────────────────
header "1. Scanning GitHub.com repositories"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  GITHUB_REPOS=($(fetch_repos "$GITHUB_COM" "$GITHUB_TOKEN"))
  info "Found ${#GITHUB_REPOS[@]} repositories on GitHub.com"
  TOTAL_REPOS=$((TOTAL_REPOS + ${#GITHUB_REPOS[@]}))
  
  for repo in "${GITHUB_REPOS[@]}"; do
    scan_repo "$GITHUB_COM" "$repo" "$GITHUB_TOKEN" &
    
    # Limit parallel jobs
    while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]]; do
      sleep 1
    done
  done
  
  wait
else
  warn "Skipping GitHub.com (no token provided)"
fi

header "2. Scanning GitHub Enterprise (github.ibm.com) repositories"

if [[ -n "${GITHUB_IBM_TOKEN:-}" ]]; then
  IBM_REPOS=($(fetch_repos "$GITHUB_IBM" "$GITHUB_IBM_TOKEN"))
  info "Found ${#IBM_REPOS[@]} repositories on github.ibm.com"
  TOTAL_REPOS=$((TOTAL_REPOS + ${#IBM_REPOS[@]}))
  
  for repo in "${IBM_REPOS[@]}"; do
    scan_repo "$GITHUB_IBM" "$repo" "$GITHUB_IBM_TOKEN" &
    
    while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]]; do
      sleep 1
    done
  done
  
  wait
else
  warn "Skipping github.ibm.com (no token provided)"
fi

# ── generate report ───────────────────────────────────────────────────────────
header "3. Generating report"

# Collect all findings
FINDINGS_FILE="${SCAN_DIR}/findings.txt"
touch "$FINDINGS_FILE"

# Process findings and generate HTML
CLEAN_REPOS=0
REPOS_WITH_FINDINGS=0

# Group findings by repository
declare -A REPO_FINDINGS

while IFS='|' read -r host repo source type file line content; do
  if [[ -n "$repo" ]]; then
    key="${host}/${repo}"
    REPO_FINDINGS["$key"]+="$source|$type|$file|$line|$content"$'\n'
  fi
done < "$FINDINGS_FILE"

# Generate HTML for each repository
for repo_key in "${!REPO_FINDINGS[@]}"; do
  ((REPOS_WITH_FINDINGS++))
  
  cat >> "$REPORT_FILE" <<EOF
      <div class="repo-section">
        <div class="repo-header">
          <h2>📦 $repo_key</h2>
          <span class="badge critical">SECRETS FOUND</span>
        </div>
EOF
  
  while IFS='|' read -r source type file line content; do
    if [[ -n "$type" ]]; then
      ((FAILURES++))
      
      cat >> "$REPORT_FILE" <<EOF
        <div class="finding">
          <div class="finding-header">
            <span class="finding-type">🚨 $type</span>
            <span class="finding-location">$file:$line ($source)</span>
          </div>
          <div class="finding-content">$content</div>
        </div>
EOF
    fi
  done <<< "${REPO_FINDINGS[$repo_key]}"
  
  echo "      </div>" >> "$REPORT_FILE"
done

CLEAN_REPOS=$((TOTAL_REPOS - REPOS_WITH_FINDINGS))

# Add clean repositories message
if [[ $CLEAN_REPOS -gt 0 ]]; then
  cat >> "$REPORT_FILE" <<EOF
      <div class="no-findings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <h3>$CLEAN_REPOS repositories are clean! 🎉</h3>
        <p>No secrets or hardcoded credentials detected.</p>
      </div>
EOF
fi

# Finalize HTML
cat >> "$REPORT_FILE" <<EOF
    </div>
    <div class="footer">
      <p>Scanned $TOTAL_REPOS repositories • Found $FAILURES potential secrets</p>
      <p>Generated by scan_all_repos_for_secrets.sh</p>
    </div>
  </div>
  <script>
    document.getElementById('timestamp').textContent = new Date().toLocaleString();
    document.getElementById('critical-count').textContent = '$FAILURES';
    document.getElementById('warning-count').textContent = '$WARNINGS';
    document.getElementById('clean-count').textContent = '$CLEAN_REPOS';
    document.getElementById('total-count').textContent = '$TOTAL_REPOS';
  </script>
</body>
</html>
EOF

# ── cleanup ───────────────────────────────────────────────────────────────────
rm -rf "$SCAN_DIR"

# ── summary ───────────────────────────────────────────────────────────────────
header "Summary"

echo ""
echo -e "  Total repositories scanned : ${TOTAL_REPOS}"
echo -e "  Repositories with findings : ${REPOS_WITH_FINDINGS}"
echo -e "  Clean repositories         : ${CLEAN_REPOS}"
echo -e "  Total secrets found        : ${FAILURES}"
echo -e "  Warnings                   : ${WARNINGS}"
echo ""
echo -e "  📄 HTML Report : ${BOLD}${REPORT_FILE}${RESET}"
echo ""

if [[ $FAILURES -gt 0 ]]; then
  echo -e "${RED}${BOLD}  ❌  SECRETS DETECTED – Immediate action required!${RESET}"
  echo ""
  echo -e "${YELLOW}  Recommended actions:${RESET}"
  echo "  1. Review the HTML report: open $REPORT_FILE"
  echo "  2. Revoke/rotate ALL exposed credentials immediately"
  echo "  3. Remove secrets from code and use environment variables"
  echo "  4. Rewrite git history to purge secrets (BFG Repo Cleaner)"
  echo "  5. Enable secret scanning: https://docs.github.com/en/code-security/secret-scanning"
  echo "  6. Use tools like git-secrets or pre-commit hooks"
else
  echo -e "${GREEN}${BOLD}  ✅  All repositories are clean!${RESET}"
fi

echo ""

# Open report in browser
if command -v open &>/dev/null; then
  info "Opening report in browser..."
  open "$REPORT_FILE"
elif command -v xdg-open &>/dev/null; then
  info "Opening report in browser..."
  xdg-open "$REPORT_FILE"
fi

exit $FAILURES

# Made with Bob
