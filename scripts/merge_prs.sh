#!/bin/bash

# Navigate to the repository root
REPO_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$REPO_ROOT" || exit 1

# Load environment variables from .env.local if it exists
if [ -f ".env.local" ]; then
    set -a
    source .env.local
    set +a
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <start_pr_number> <end_pr_number>"
    exit 1
fi

START_PR=$1
END_PR=$2

echo "Starting batch merge for PRs #$START_PR to #$END_PR..."
echo "----------------------------------------------------"

for pr in $(seq $START_PR $END_PR); do
    echo "Attempting to merge PR #$pr..."
    
    # First ensure the PR is not in a Draft state
    gh pr ready "$pr" >/dev/null 2>&1 || true

    # We use --squash to squash commits (common for AI generated PRs)
    # We use -d to delete the remote branch after merging
    gh pr merge "$pr" --squash -d
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully merged PR #$pr"
    else
        echo "❌ Failed to merge PR #$pr (it might already be merged or closed)"
    fi
    echo "----------------------------------------------------"
done

echo "Batch merge complete!"
echo "Run 'git pull origin main' to sync your local branch."
