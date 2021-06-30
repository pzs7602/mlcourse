## README.md

### object tracking
https://github.com/xingyizhou/CenterTrack/
install DCNv2 and insert python package search path in all(2files) .py files:
```
import sys
sys.path.append("/home/pzs/pzs/CenterTrack/src/lib/model/networks/DCNv2")
```
test:
```
python3 demo.py tracking --load_model ./coco_tracking.pth --demo ../videos/nuscenes_mini.mp4 

```
# push local directory to github

create a project a github.com
the files larger then 100M cannt be pushed to remote repository

echo "# tensorrt_demos" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:pzs7602/tensorrt_demos.git
git push -u origin main

the last command will prompt you for the username/password in github.com

or if files modified, commit:
use git add/rm <file> to add or remove modified files from local repository to be commited
git commit -m "files renew commit"
git push -u origin main

if comamnd: 
```
git push -u origin main  
```
error:
The authenticity of host 'github.com (52.74.223.119)' can't be established.
RSA key fingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'github.com,52.74.223.119' (RSA) to the list of known hosts.
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

this may caused by missing publickey in ~/.ssh , do:
```
ssh-keygen -t rsa # this create ~/.ssh/id_rsa.pub
cat ~/.ssh/id_rsa.pub
# from https://github.com/pzs7602 , in settings-> SSH and GPG keys-> New SSh key and paste the id_rsa.pub key
```

### get opencv build information
```
import cv2
print(cv2.getBuildInformation())
```
