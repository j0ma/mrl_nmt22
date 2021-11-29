#!/bin/bash

# source: https://castorfou.github.io/guillaume_blog/blog/git-ignore-large-files.html

# update gitignore_bigfiles
find . -size +100M -not -path "./.git*"| sed 's|^\./||g' | cat > .gitignore_bigfiles

# create gitignore as concat of gitingore_static and gitignore_bigfiles
cat .gitignore_static .gitignore_bigfiles > .gitignore

# print content of .gitignore_bigfiles
cat .gitignore_bigfiles
