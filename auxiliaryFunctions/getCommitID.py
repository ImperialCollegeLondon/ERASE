############################################################################
#
# Imperial College London, United Kingdom
# Multifunctional Nanomaterials Laboratory
#
# Project:  ERASE
# Year:     2020
# Python:   Python 3.7
# Authors:  Ashwin Kumar Rajagopalan (AK)
#
# Purpose:
# Generates a short SHA key of the git commit id in the current branch and
# repository
#
# Last modified:
# - 2020-10-19, AK: Initial creation
#
# Input arguments:
# - N/A
#
# Output arguments:
# - short_sha: Short git commit ID 
#
############################################################################

def getCommitID():
    # Use gitpython to get the git information of the current repository
    import git
    repo = git.Repo(search_parent_directories=True)
    # Get the simple hashing algorithm tag (SHA)
    sha = repo.head.commit.hexsha
    # Parse the first six characters of the sha
    short_sha = repo.git.rev_parse(sha, short=7)
    
    # Return the git commit id
    return short_sha