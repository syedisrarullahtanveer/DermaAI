import os
import subprocess
import sys

# Helper to run a git command
def git_cmd(args, cwd=None):
    result = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print('Git command failed:', 'git', ' '.join(args))
        print('stderr:', result.stderr)
        sys.exit(1)
    return result.stdout.strip()

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Ensure utils package exists
utils_init = os.path.join(repo_path, 'src', 'utils', '__init__.py')
if not os.path.exists(utils_init):
    os.makedirs(os.path.dirname(utils_init), exist_ok=True)
    open(utils_init, 'w').write('# utils package\n')

batch_file = os.path.join(repo_path, 'src', 'utils', 'batch_helpers.py')
# Start with a header comment
if not os.path.exists(batch_file):
    with open(batch_file, 'w') as f:
        f.write('# Batch helper functions generated for large commit batch\n\n')

# Function to append helper function
def add_helper(i):
    func_name = f'helper_{i}'
    code = f"def {func_name}(x: int) -> int:\n    """Add {i} to x."""\n    return x + {i}\n\n"
    with open(batch_file, 'a') as f:
        f.write(code)
    git_cmd(['add', batch_file], cwd=repo_path)
    commit_msg = f'feat(utils): add {func_name} function'
    git_cmd(['commit', '-m', commit_msg], cwd=repo_path)

# Function to add unit test for a helper
def add_helper_test(i):
    test_filename = os.path.join(repo_path, 'tests', f'test_helper_{i}.py')
    func_name = f'helper_{i}'
    content = f"""import unittest\nimport sys, os\nsys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))\nfrom src.utils.batch_helpers import {func_name}\n\nclass Test{func_name.capitalize()}(unittest.TestCase):\n    def test_addition(self):\n        self.assertEqual({func_name}(10), 10 + {i})\n\nif __name__ == '__main__':\n    unittest.main()\n"""
    # Note: extra parenthesis typo fix below
    content = content.replace('))', ')')
    with open(test_filename, 'w') as f:
        f.write(content)
    git_cmd(['add', test_filename], cwd=repo_path)
    commit_msg = f'test(utils): add unit test for {func_name}'
    git_cmd(['commit', '-m', commit_msg], cwd=repo_path)

# Main loop: 50 helpers and 50 tests = 100 commits
for idx in range(1, 51):
    add_helper(idx)
    # after each helper commit, optionally run tests every 20 commits
    total_commits = int(git_cmd(['rev-list', '--count', 'HEAD'], cwd=repo_path))
    if total_commits % 20 == 0:
        # run existing test suite to ensure no regressions
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'], cwd=repo_path)
    add_helper_test(idx)
    total_commits = int(git_cmd(['rev-list', '--count', 'HEAD'], cwd=repo_path))
    if total_commits % 20 == 0:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'], cwd=repo_path)

print('Batch of 100 commits completed.')
