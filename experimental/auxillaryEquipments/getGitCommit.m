% Read out the git commit ID of the git repository to which the current directory
% (or the directory defined in varargin{1}) belongs. Return the ID as a string.

function commitId = getGitCommit(varargin)
% Identify target directory
if ~isempty(varargin)
    oldDir = cd;
    targetDir = varargin{1};
    cd(targetDir);
end
% Get short version of git commit ID
[status,cmdout] = system('git rev-parse HEAD');
if status == 0
    % Command was successful
    commitId = cmdout(1:7);
else
    commitId = [];
end
% cd back to initial directory
if exist('oldDir','var')
    cd(oldDir);
end
end

