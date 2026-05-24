
# Space Data 3 Project - IMAGE DENOISING

### Setup
Once on the Euler cluster, run the modified startup script to setup the environment <br>  (on initial setup also run ```chmod +x startup.sh```):
```
source ./startup.sh 
```


### File Structure
* data - stores train data, val data (Contents not pushed to git, manually add the train and test data from moodle)
* src - stores implementation
* graphics - store all plots here [WITH UNIQUE NAMING FOR REPORT]


### Implementation Ideas

* U-Net 

* U-Net with skip connections

* U-Net with attention layer

### To-Do's
* Split Train data in Train and Test split
* Data loader
* Model implementation
* Training Loop implementation
* Hyperparameter tuning
* Cross validation implementation
* Write report