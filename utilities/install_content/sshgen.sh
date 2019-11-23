echo First, whats your full name?
read name
echo It\'s nice to meet you $name, my name is kieran\'s-super-awesome-script, what is your github email?
read email

ssh-keygen -o -t rsa -b 4096 -C "$email"
echo
echo "https://github.com/settings/ssh"
read -p "Press [Enter] key when you have opened the above link..."
echo
cat ~/.ssh/id_rsa.pub
echo
echo "The sshkey is shown above"
read -p "Press [Enter] key to start backup once you have added it to your keys page, if you do not then things will not work..."
git remote set-url origin git@github.com:cv-core/PerceptionCV.git 
git config --global user.name "$name"
git config --global user.email "$email"
git config --global core.editor vim
git config --global diff.tool meld
git config --global merge.tool meld
echo -e '[alias] \n        lg1 = log --graph --abbrev-commit --decorate --format=format:'"'"'%C(bold blue)%h%C(reset) - %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) %C(dim white)- %an%C(reset)%C(bold yellow)%d%C(reset)'"'"' --all \n        lg2 = log --graph --abbrev-commit --decorate --format=format:'"'"'%C(bold blue)%h%C(reset) - %C(bold cyan)%aD%C(reset) %C(bold green)(%ar)%C(reset)%C(bold yellow)%d%C(reset)%n'"'""'"'%C(white)%s%C(reset) %C(dim white)- %an%C(reset)'"'"' --all' >> ~/.gitconfig

