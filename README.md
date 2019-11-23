#### Setting Up Google Cloud Client on Ubuntu 18.04
First, get your personal Gmail or MIT GApp account added to the 'MIT Driverless' project. You can ask the current CV sub-team lead (Kieran, kstrobel@mit.edu) to do this for you. Then run:
```
sudo snap install google-cloud-sdk --classic
gcloud auth login # and log into the Google account with which you will use GCP
gcloud config set project mitdriverless # mitdriverless is the current project ID
gcloud init
```
This will install and authenticate the client on your local machine. Now to create a new instance, please go to "VM instances" page of Google Cloud Plateform: `https://console.cloud.google.com/compute/instances?project=mitdriverless&duration=PT1H`, and then click "CREATE INSTANCE"

<img src="https://user-images.githubusercontent.com/22118253/57960934-b6b40c00-78d9-11e9-8ebb-cff5564850b5.png" width="600">

Then click "New VM instance from template" and choose "cv-template-1xp100-v3" and click "Continue"

<img src="https://user-images.githubusercontent.com/22118253/57960969-eb27c800-78d9-11e9-9432-4d0283c3ebe0.png">

Name the instance in your way and scroll down and click "Create"

After your new instance has been succussfully created, enter the code below in your terminal:

```
gcloud compute --project "mitdriverless" ssh --zone "<zone>" "<instance name>"
```
where instance name is one of instance-04, instance-05, etc.

### Setting up repository on new instance

```
git clone https://github.com/DUT-Racing/DUT18D_PerceptionCV.git
cd DUT18D_PerceptionCV/vectorized-yolov3
python3 setup.py install
sudo rm /usr/bin/pip
pip2 install onnx==1.1.1
```

After the step above, you are all good to start adventure in the CV world! 

### Setting up SSH keys on the Instance
```
ssh-keygen -t rsa -b 4096 -C "<user@gmail.com>" # all empty settings
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
```
And copy paste the output to new key on `https://github.com/settings/keys`!

### TMUX
Using tmux is encouraged to avoid killing jobs on log-off:
```
tmux # to create a new session
```
Hit Ctrl+b and then d to leave the session.

To look at current live sessions:
```
tmux list-sessions # or tmux ls
```

To enter the session:
```
tmux attach-session -t <session number> # or tmux a -t <session number>
```

To kill certain tmux session:
```
tmux kill-session -t <session number>
```

To kill all tmux session:
```
pkill -f tmux
```

To scroll inside a session: Ctrl-b then [, then q to exit scroll mode.

### Errors

ERROR: (gcloud.compute.ssh) [/usr/bin/ssh] exited with return code [255].

solution: go to your .ssh and ditch the config file and all the google files, then run gcloud compute config-ssh and you should be good

How to setup a Desktop environment: 

### Graphical Interface

Follow instructions here to set up a VM instance with VNC/Xorg for a GNOME Desktop:
https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781788474221/1/ch01lvl1sec15/installing-and-configuring-ubuntu-desktop-for-google-cloud-platform

For convenience, the instructions you might need to repeat from the link above are listed here:
Login to the VM and run:
`vncserver -geometry 1920×1200`
If you need to kill `vncserver`:
`vncserver -kill :1`
Then, establish a connection to the gcloud instance in another terminal window:
`gcloud compute --project "mitdriverless" ssh --zone "us-east1-b" "cv-research-1" --ssh-flag "-L 5901:localhost:5901"`
Finally, open up VNC client on your host machine and connect via `localhost::5091`
