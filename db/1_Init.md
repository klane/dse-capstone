## Launching EC2 instance to populate database

- Launched Amazon Linux 2 AMI instance using t2-micro.
- Install Anaconda locally:

    ```
    wget wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    bash ./Anaconda3-2019.03-Linux-x86_64.sh
    ```

- The EC2 instance will be stopped (not terminated), so this step only needs to be done once.
- Log back in, setting ssh tunnel to jupyter listening port and start jupyter server on that port:

    ```
    ssh -i myawscert.pem -L 8888:localhost:8888 ec2-user@ec2-54-213-238-126.us-west-2.compute.amazonaws.com

    jupyter notebook --port 8888 &> nblog &
    ```