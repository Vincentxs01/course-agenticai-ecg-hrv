# Agentic AI Course - Student Submissions Repository

This repository contains all student submissions for the **Agentic AI** course (代理式人工智慧) at National Cheng Kung University.

---

## IMPORTANT NOTICES

### License Requirements

All submitted material **must** be licensed under one of the following:

- **Apache License 2.0** (for code)
- **CC-BY-4.0** (Creative Commons Attribution 4.0, for documentation and data)

You may **only** submit material that you have the right to make public under these licenses. Each submission folder and file **must** clearly indicate its license.

### Privacy and Data Protection

**NO PERSONALLY IDENTIFIABLE INFORMATION (PII) MAY BE INCLUDED.**

This includes but is not limited to:
- Names linked to data (other than author attribution, which naturally must be included)
- Student IDs, national IDs, or any identification numbers
- Photos of identifiable individuals
- Health data linked to identifiable persons
- Contact information of subjects
- Any information that could link collected data to a specific person

**Violation of this policy may result in immediate removal of your submission and disciplinary action.**

---

## Repository Structure

```
student-material-ecg-hrv/
│
├── README.md                      # This file
├── Syllabus_....md                # Course syllabus (reference)
│
├── case-brief-individual/         # Individual case briefs (Markdown)
├── reflection-group/              # Group reflection documents (Markdown)
├── report-individual/             # Individual technical reports (Markdown)
│
├── data-group/                    # Group data submissions (folder per group)
│   └── 2026-Chen-Lin-Wang-data/   # Example: git submodule
│
├── project-code-group/            # Group project code (folder per group)
│   └── 2026-Chen-Lin-Wang-code/   # Example: git submodule
│
├── tests-group/                   # Group test cases and results document (Markdown)
│
├── slides-demonstration-group/    # Group presentation slides (Beamer .tex)
└── system-design-group/           # Group system design diagrams (draw.io .drawio)
```

### Example Submodules

The example data and code repositories are included as git submodules:

| Submodule | HTTPS URL | SSH URL |
|-----------|-----------|---------|
| `data-group/2026-Chen-Lin-Wang-data` | `https://bitbucket.org/nordlinglab/nordlinglab-course-agenticai-ecg-hrv-example-data.git` | `git@bitbucket.org:nordlinglab/nordlinglab-course-agenticai-ecg-hrv-example-data.git` |
| `project-code-group/2026-Chen-Lin-Wang-code` | `https://bitbucket.org/nordlinglab/nordlinglab-course-agenticai-ecg-hrv-example-code.git` | `git@bitbucket.org:nordlinglab/nordlinglab-course-agenticai-ecg-hrv-example-code.git` |

**For maintainers:** To set up these submodules, run from the main course repository:
```bash
# From nordlinglab-course-agenticai/ directory
chmod +x src/setup_example_submodules.sh
./src/setup_example_submodules.sh --help    # See all options
./src/setup_example_submodules.sh           # Use defaults
./src/setup_example_submodules.sh --register  # Initialize, push, and register
```

---

## File Naming Convention

### IMPORTANT: ASCII Characters Only

**Use only ASCII characters in file and folder names. No Chinese characters, spaces, or special characters.**

### Individual Submissions

Format: `YYYY-FamilyName-FirstName.md`

Examples:
- `2026-Chen-Wei.md`
- `2026-Lin-MeiLing.md`
- `2026-Wang-XiaoMing.md`

### Group Submissions

Format: `YYYY-FamilyName1-FamilyName2-FamilyName3/` (folder) or `YYYY-FamilyName1-FamilyName2-FamilyName3.ext` (file)

List all group members' family names in alphabetical order.

Examples:
- `2026-Chen-Lin-Wang-code` (folder for group code submissions)
- `2026-Chen-Lin-Wang.tex` (Beamer slides)
- `2026-Chen-Lin-Wang.drawio` (system design)

---

## Submission Format Requirements

| Submission Type | Format | Location |
|-----------------|--------|----------|
| Case Brief (Individual) | Markdown `.md` | `case-brief-individual/` |
| Reflection (Group) | Markdown `.md` | `reflection-group/` |
| Technical Report (Individual) | Markdown `.md` | `report-individual/` |
| Data (Group) | Folder with data files incl. `README.md` | `data-group/` |
| Project Code (Group) | Folder with code incl. `README.md` | `project-code-group/` |
| Test Cases (Group) | Markdown `.md` with reference to `tests/` in project code | `tests-group/` |
| Presentation Slides (Group) | Beamer `.tex` (NordlingLab 16:9 template) | `slides-demonstration-group/` |
| System Design (Group) | draw.io `.drawio` XML (UML standard) | `system-design-group/` |

See the `README.md` in each folder for detailed requirements, grading criteria, and examples.

---

## Git Guide for Beginners

If you have never used Git before, follow these steps carefully.

### Step 1: Install Git

**macOS:**
```bash
# Open Terminal and run:
xcode-select --install
```

**Windows:**
Download and install from: https://git-scm.com/download/win

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

### Step 2: Configure Git

Open your terminal and set your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 3: Create a Bitbucket Account and Workspace

**You need BOTH a Bitbucket account AND a workspace to fork repositories.**

#### If you don't have a Bitbucket account:

> **WARNING: Do NOT use "Continue with Google" to sign up!**
>
> If you sign up with Google SSO, you won't have a password, which means you cannot use HTTPS to clone/push/pull repositories. You would need to set up SSH keys instead (more complicated for beginners).
>
> **Recommended:** Sign up with email and create a password. This allows you to use HTTPS URLs which are simpler.

1. Go to https://bitbucket.org/account/signup/
2. Click **"Sign up with email"** (NOT "Continue with Google")
3. Enter your email and create a **password** (you can use your university email)
4. **Important:** When prompted, create a **personal workspace** and give it a unique easy to remember name - this is where your fork will live. Other people find your forked repository in `https://bitbucket.org/THE_WORKSPACE_NAME_YOU_PICKED`

### Step 4: Fork the Repository to Your Own Workspace

Since you have **read-only access** to the original repository, you must create your own copy (fork) in **your personal workspace**:

1. Log in to [Bitbucket](https://bitbucket.org/)
2. Go to [this repository](https://bitbucket.org/nordlinglab/nordlinglab-course-agenticai-ecg-hrv/)
3. Click the **"..."** button (three dots) in the upper right corner
4. Select **"Fork this repository"**
5. **IMPORTANT - Configure the fork:**

   - **Workspace:** Select **THE_WORKSPACE_NAME_YOU_PICKED** (not `nordlinglab` - you don't have write access there)
   - **Project:** Select or create a project in your workspace (e.g., "My Projects")
   - **Repository name:** Give it a name (e.g., `agenticai-ecg-hrv-submissions`)
   - **Access level:** Leave unchecked (public) so group members can access it, OR check "Private" and manually invite group members to the workspace

6. Click **"Fork repository"**

You now have your own copy at `https://bitbucket.org/THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME/` where you can make changes.

**Common Error A:** If you see "Access denied" or "You don't have permission", you likely tried to fork into the `nordlinglab` workspace. Go back and select YOUR OWN workspace in the Workspace dropdown.

**Common Error B:** If you have an account but NO workspace appears in the fork dialog:

This happens when you signed up via Atlassian/Google SSO without creating a Bitbucket workspace. You must create one:

1. Go to https://bitbucket.org/account/workspaces/
2. Click **"Create workspace"**
3. Enter a **Workspace name** (e.g., your username, name, or nickname)
4. The **Workspace ID** will be auto-generated (this becomes part of your repository URLs)
5. Click **"Create"**

After creating a workspace, go back to the repository and try forking again - your new workspace should now appear in the dropdown.

### Step 5: Clone Your Fork

Use either HTTPS (simpler, requires password) or SSH (requires key setup):

```bash
# Navigate to where you want to store the project
cd ~/Documents

# Clone YOUR fork with submodules (not the original repository)
# Option 1: HTTPS (recommended for beginners - uses your Atlassian password)
git clone --recursive https://bitbucket.org/THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git

# Option 2: SSH (if you have SSH keys set up)
git clone --recursive git@bitbucket.org:THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git

# Enter the project directory
cd YOUR_FORK_NAME
```

**Note:** The `--recursive` flag initializes all git submodules (example data and code repositories).

If you already cloned without `--recursive`, initialize submodules with:

```bash
git submodule update --init --recursive
```

### Step 6: Create Your Submission

1. Navigate to the appropriate folder
2. Create your file following the naming convention
3. Add required content (see folder [README](./case-brief-individual/README.md) for requirements)

Example for case brief:
```bash
cd case-brief-individual

# Create your file (use a text editor)
# File must be named: YYYY-YourFamilyName-YourFirstName.md
```

### Step 7: Stage and Commit Your Changes

```bash
# Check what files have changed
git status

# Stage your new file(s)
# Option 1: If you are in the folder case-brief-individual
git add YYYY-YourFamilyName-YourFirstName.md
# Option 2: If you are in the root of the repository
git add case-brief-individual/YYYY-YourFamilyName-YourFirstName.md

# Commit with a descriptive message
git commit -m "Add case brief for YourName"
```

### Step 8: Push to Your Fork

```bash
git push origin main
```

### Step 9: Create a Pull Request

1. Go to **your fork** on Bitbucket
2. Click **"Create pull request"** (or find it under the **"..."** button (three dots) in the upper right corner)
3. Set:

   - **Source:** your fork's `main` branch
   - **Destination:** the original repository's `main` branch

4. Add a title: `Submission: YYYY-YourFamilyName-YourFirstName - [type]`
5. Add description of what you're submitting
6. Click **"Create pull request"**

The TA or group member whose repository you forked will review your submission and merge it if it meets the requirements.

---

## Common Git Commands Reference

| Command | Description |
|---------|-------------|
| `git status` | Show changed files |
| `git add <file>` | Stage a file for commit |
| `git add .` | Stage all changed files |
| `git commit -m "message"` | Commit staged changes |
| `git push` | Upload commits to remote |
| `git pull` | Download updates from remote |
| `git log` | View commit history |
| `git diff` | Show unstaged changes |

---

## Updating Your Fork

If the original repository is updated, sync your fork:

```bash
# Add the original repo as "upstream" (do this once)
# Option 1: HTTPS
git remote add upstream https://bitbucket.org/nordlinglab/nordlinglab-course-agenticai-ecg-hrv.git
# Option 2: SSH
git remote add upstream git@bitbucket.org:nordlinglab/nordlinglab-course-agenticai-ecg-hrv.git

# Or if you forked a group member's repository:
# Option 1: HTTPS
git remote add upstream https://bitbucket.org/THE_WORKSPACE_NAME_YOUR_GROUP_MEMBER_PICKED/YOUR_GROUP_MEMBERS_FORK_NAME.git
# Option 2: SSH
git remote add upstream git@bitbucket.org:THE_WORKSPACE_NAME_YOUR_GROUP_MEMBER_PICKED/YOUR_GROUP_MEMBERS_FORK_NAME.git

# Fetch updates from original (including submodules)
git fetch upstream

# Merge updates into your branch
git merge upstream/main

# Update submodules to latest
git submodule update --init --recursive

# Push updates to your fork
git push origin main
```

---

## Reviewing Pull Requests

This section explains how to review code changes from your group members.

### In Bitbucket Web Interface

#### Viewing a Pull Request

1. Go to the repository on Bitbucket
2. Click **"Pull requests"** in the left sidebar
3. Click on the pull request you want to review
4. You'll see:

   - **"Overview"** tab: Description, commits, and status
   - **"Files changed"** tab: All changed files with line-by-line changes
   - **"Commits"** tab: Individual commits in the PR

#### Reviewing Changes

1. Click on the **"Files changed"** tab
2. Changes are shown with:

   - **Green background:** Added lines
   - **Red background:** Removed lines

3. Click on any line number and the **+** that appears to add a comment

#### Adding Comments and Requesting Changes

1. Click on a line number and the **+** that appears in the diff view on the **"Files changed"** tab
2. Type your comment (you can use Markdown)
3. Click **"Overview"** tab and write a general comment in the Activity

#### Approving or Declining a Pull Request

1. After reviewing all changes, click the **"Approve"** button (green checkmark) if the changes look good (you still need to merge)
2. Or click **"Request changes"** (orange icon) if modifications are needed
3. To decline entirely, click **"Decline"** from the **"..."** menu

#### Resolving "This pull request is X commits behind"

**Problem:** You see a warning like:
> "This pull request is 1 commit behind nordlinglab/nordlinglab-course-agenticai-ecg-hrv:main"

**What this means:** The target branch has new commits that aren't in your PR branch. You need to update your branch.

**Solution (for the author of the pull request):**

```bash
# In your local repository
cd YOUR_FORK_NAME

# Fetch the latest from upstream
git fetch upstream

# Merge upstream changes into your branch
git merge upstream/main

# If there are conflicts, resolve them in your editor, then:
git add .
git commit -m "Merged upstream changes"

# Push the updated branch
git push origin main
```

**Alternative - Rebase (cleaner history):**

```bash
git fetch upstream
git rebase upstream/main

# If conflicts occur, resolve them, then:
git add .
git rebase --continue

# Push the updated branch
git push origin main
```

After pushing, the pull request will automatically update and the warning should disappear.

#### Merging a Pull Request

1. Once approved and all checks pass, click **"Merge"** from the **"..."** menu to merge the branch into `nordlinglab/nordlinglab-course-agenticai-ecg-hrv:main`
2. Choose merge strategy:

   - **Merge commit:** Preserves all commits
   - **Squash:** Combines all commits into one
   - **Fast-forward:** Only if no divergence (cleanest history so recommended)

3. Click **"Merge"** to confirm

---

### In the Terminal

#### Setting Up Better Diff Tools

The default `git diff` output can be hard to read. Here are better alternatives:

**Option 1: Colored diff with `colordiff`**

```bash
# Install colordiff
# macOS
brew install colordiff

# Ubuntu/Debian
sudo apt install colordiff

# Usage: pipe git diff through colordiff
git diff | colordiff

# Or set it as default pager
git config --global core.pager "colordiff | less -R"
```

**Option 2: `git-delta` (Modern, feature-rich)**

```bash
# Install delta
# macOS
brew install git-delta

# Ubuntu/Debian (download from GitHub releases)
# https://github.com/dandavison/delta/releases

# Windows (with scoop)
scoop install delta

# Configure git to use delta
git config --global core.pager delta
git config --global interactive.diffFilter "delta --color-only"
git config --global delta.navigate true
git config --global delta.side-by-side true
```

Delta features:

- Syntax highlighting
- Side-by-side view
- Line numbers
- Word-level diff highlighting

#### Reviewing a Pull Request Locally

```bash
# Fetch the pull request branch from your group member's fork
# First, add their fork as a remote (do this once)
git remote add groupmate https://bitbucket.org/THEIR_WORKSPACE/THEIR_FORK.git

# Fetch their branches
git fetch groupmate

# View the diff between your main and their branch
# Option 1: built-in diff
git diff main..groupmate/main
# Option 2: delta for better visualization
git diff main..groupmate/main | delta

# View only file names that changed
git diff --name-only main..groupmate/main

# View diff statistics (lines added/removed per file)
git diff --stat main..groupmate/main
```

#### Checking Out a Pull Request Branch for Testing

```bash
# Create a local branch from their pull request
git checkout -b review-groupmate groupmate/main

# Run tests, check the code, etc.
# ...

# When done, switch back to your branch
git checkout main

# Delete the review branch
git branch -d review-groupmate
```

#### Viewing Commit History

```bash
# See commits in the PR
git log main..groupmate/main --oneline

# See commits with full details
git log main..groupmate/main

# See commits as a graph
git log main..groupmate/main --oneline --graph
```

#### Adding Comments via Command Line

While you can't add pull request comments directly from the terminal like GitHub's `gh` CLI, you can:

**1. Create a review file locally:**

```bash
# Save diff to a file for adding your comments
git diff main..groupmate/main > review_notes.txt
# Edit review_notes.txt with your comments, then share via pull request comment
```

**2. Use the Bitbucket REST API with curl:**

```bash
# Add a comment to a PR (requires app password)
curl -u "your_email:app_password" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"content": {"raw": "Your comment here"}}' \
  "https://api.bitbucket.org/2.0/repositories/WORKSPACE/REPO/pullrequests/PR_ID/comments"
```

**3. Use a Bitbucket CLI tool (optional):**

Note: Unlike GitHub's official `gh` CLI, Bitbucket doesn't have an official CLI. Third-party options exist but are limited.

```bash
# Best practice: Use uv (fastest, isolated environment)
# First install uv if you don't have it:
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or with Homebrew
brew install uv

# Then install bitbucket-cli in isolated environment
uv tool install bitbucket-cli

# Alternative: Use pipx (also isolated)
brew install pipx    # macOS
pipx install bitbucket-cli

# NOT recommended: Direct pip install (breaks PEP 668 on modern systems)
# pip install bitbucket-cli  # Avoid this
```

For most users, the Bitbucket web interface is the easiest way to manage pull request comments.

---

### In VS Code

#### Built-in Git Diff

1. Open VS Code in your repository folder
2. Click the **Source Control** icon (branch icon) in the left sidebar, or press `Ctrl+Shift+G` / `Cmd+Shift+G`
3. Changed files appear in the sidebar
4. Click any file to see a side-by-side diff view

#### Comparing Branches

1. Open the Command Palette: `Ctrl+Shift+P` / `Cmd+Shift+P`
2. Type **"Git: Checkout to..."** and select the branch to compare
3. Or use the **GitLens** extension for more powerful comparisons

#### Using GitLens Extension (Recommended)

**Install GitLens:**

1. Go to Extensions (`Ctrl+Shift+X` / `Cmd+Shift+X`)
2. Search for **"GitLens"**
3. Click **Install**

**Compare branches with GitLens:**

1. Click the **GitLens** icon in the sidebar
2. Expand **"Compare"** section
3. Click **"Compare References..."**
4. Select your branch (e.g., `main`)
5. Select the branch to compare against (e.g., `groupmate/main`)
6. GitLens shows:

   - List of changed files
   - Number of commits difference
   - Click any file to see the diff

**Review inline blame:**

- GitLens shows who changed each line and when
- Hover over any line to see the commit details

#### Fetching and Reviewing Remote Branches in VS Code

1. Open the Command Palette: `Ctrl+Shift+P` / `Cmd+Shift+P`
2. Type **"Git: Fetch From..."**
3. Select the remote (e.g., `groupmate`)
4. Now you can compare against `groupmate/main`

#### VS Code Pull Request Extension (for GitHub/GitLab)

Note: For Bitbucket, use the **Atlassian for VS Code** extension:

1. Install **"Atlassian for VS Code"** from Extensions
2. Sign in to your Atlassian account
3. You can view and manage Bitbucket PRs directly in VS Code

---

## Troubleshooting

### "Access denied" when forking

**Problem:** You see "Access denied" or "You don't have permission to create a repository in this workspace" when trying to fork.

**Solution:** You're trying to fork into the `nordlinglab` workspace, which you don't have write access to. When forking:

1. Look for the **"Workspace"** dropdown in the fork dialog
2. Change it from `nordlinglab` to **your personal workspace** (your username)

### No workspace available in fork dialog

**Problem:** The Workspace dropdown is empty or only shows `nordlinglab` (which you can't use).

**Cause:** You signed up for Bitbucket via Atlassian ID or Google SSO, but never created a personal workspace. Bitbucket accounts created this way don't automatically get a workspace.

**Solution:** Create a workspace first:

1. Go to https://bitbucket.org/account/workspaces/
2. Click **"Create workspace"**
3. Enter a name (e.g., your username, name, or nickname)
4. Click **"Create"**
5. Return to the repository and try forking again

### "Permission denied" when pushing

You're trying to push to the original repository instead of your fork. Check your remote URL:

```bash
git remote -v
```

If it shows `nordlinglab` in the URL, you cloned the original instead of your fork. Fix it:

```bash
# Option 1: HTTPS (recommended for beginners)
git remote set-url origin https://bitbucket.org/THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git

# Option 2: SSH (if you have SSH keys set up)
git remote set-url origin git@bitbucket.org:THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git
```

### Merge conflicts

If your pull request shows conflicts:

1. Pull the latest changes: `git pull upstream main`
2. Resolve conflicts in your editor
3. Commit the resolved files
4. Push again

### Wrong file name

If you named your file incorrectly:
```bash
# Rename the file
git mv old-name.md YYYY-CorrectName-Format.md
git commit -m "Fix file naming"
git push
```

### Cannot git push/pull - "Password required" but signed up with Google

**Problem:** You signed up with "Continue with Google" and now git asks for a password when you try to push/pull, but you don't have one.

**Two solutions:**

#### Solution A: Create an Atlassian password (Recommended for beginners)

1. Go to https://id.atlassian.com/manage-profile/security
2. Click **"Create password"** or **"Set password"**
3. Enter a new password and confirm it
4. Now you can use HTTPS URLs with your email and this new password

When git asks for credentials:

- **Username:** Your Atlassian email address
- **Password:** The password you just created

#### Solution B: Set up SSH keys (More advanced but more convenient long-term)

SSH keys let you authenticate without entering a password each time. Follow the instructions below for your operating system.

---

### Setting up SSH Keys for Bitbucket

#### macOS

**Step 1: Check for existing SSH keys**

```bash
ls -la ~/.ssh
```
If you see `id_ed25519` and `id_ed25519.pub` (or `id_rsa` and `id_rsa.pub`), you already have keys. Skip to Step 3.

**Step 2: Generate a new SSH key**

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

- Press Enter to accept the default file location
- Enter a passphrase (optional but recommended) or press Enter for no passphrase

**Step 3: Start the SSH agent and add your key**

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**Step 4: Copy your public key**

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```
This copies the key to your clipboard.

**Step 5: Add the key to Bitbucket**

1. Go to https://bitbucket.org/account/settings/ssh-keys/
2. Click **"Add key"**
3. Give it a label (e.g., "My MacBook")
4. Paste the key (Cmd+V)
5. Click **"Add key"**

**Step 6: Test the connection**

```bash
ssh -T git@bitbucket.org
```
You should see: "authenticated via ssh key" or similar.

**Step 7: Update your repository to use SSH**

```bash
cd YOUR_FORK_NAME
git remote set-url origin git@bitbucket.org:THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git
```

---

#### Windows

**Step 1: Open Git Bash**

If you installed Git for Windows, you have Git Bash. Search for "Git Bash" in the Start menu.

**Step 2: Check for existing SSH keys**

```bash
ls -la ~/.ssh
```
If you see `id_ed25519` and `id_ed25519.pub`, you already have keys. Skip to Step 4.

**Step 3: Generate a new SSH key**

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

- Press Enter to accept the default file location
- Enter a passphrase (optional) or press Enter for no passphrase

**Step 4: Start the SSH agent and add your key**

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**Step 5: Copy your public key**

```bash
clip < ~/.ssh/id_ed25519.pub
```
This copies the key to your clipboard.

**Step 6: Add the key to Bitbucket**

1. Go to https://bitbucket.org/account/settings/ssh-keys/
2. Click **"Add key"**
3. Give it a label (e.g., "My Windows PC")
4. Paste the key (Ctrl+V)
5. Click **"Add key"**

**Step 7: Test the connection**

```bash
ssh -T git@bitbucket.org
```
You should see: "authenticated via ssh key" or similar.

**Step 8: Update your repository to use SSH**

```bash
cd YOUR_FORK_NAME
git remote set-url origin git@bitbucket.org:THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git
```

---

#### Linux (Ubuntu/Debian)

**Step 1: Check for existing SSH keys**

```bash
ls -la ~/.ssh
```
If you see `id_ed25519` and `id_ed25519.pub`, you already have keys. Skip to Step 3.

**Step 2: Generate a new SSH key**

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```
- Press Enter to accept the default file location
- Enter a passphrase (optional but recommended) or press Enter for no passphrase

**Step 3: Start the SSH agent and add your key**

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**Step 4: Copy your public key**

```bash
# Install xclip if not installed
sudo apt install xclip

# Copy to clipboard
xclip -selection clipboard < ~/.ssh/id_ed25519.pub
```

Or manually display and copy:
```bash
cat ~/.ssh/id_ed25519.pub
```
Then select and copy the output.

**Step 5: Add the key to Bitbucket**

1. Go to https://bitbucket.org/account/settings/ssh-keys/
2. Click **"Add key"**
3. Give it a label (e.g., "My Linux PC")
4. Paste the key (Ctrl+Shift+V in terminal, or Ctrl+V in browser)
5. Click **"Add key"**

**Step 6: Test the connection**

```bash
ssh -T git@bitbucket.org
```
You should see: "authenticated via ssh key" or similar.

**Step 7: Update your repository to use SSH**

```bash
cd YOUR_FORK_NAME
git remote set-url origin git@bitbucket.org:THE_WORKSPACE_NAME_YOU_PICKED/YOUR_FORK_NAME.git
```

---

## Getting Help

- **Git documentation:** https://git-scm.com/doc
- **Bitbucket tutorials:** https://www.atlassian.com/git/tutorials
- **TA contact:** See course syllabus for TA information

---

## License and Disclaimer

**Author:** Torbjörn E. M. Nordling, PhD

Unless otherwise specified, content in this repository is licensed under:

- **Documentation:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Source Code:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

See individual submissions for specific license declarations.

THIS DOCUMENT AND ALL ASSOCIATED CODE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THIS DOCUMENT OR THE USE OR OTHER DEALINGS IN THIS DOCUMENT.
