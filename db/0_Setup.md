## Setting up Postgres SQL on AWS

Description of steps taken to set up database and allow external connections.

### Launch Database using RDS

- mostly self-explanatory.   
- I chose free tier as configuration for now (t2-micro)
- Up to 20 Gb to stay in free tier, but this can be adjusted as needed.
- Allowed IAM access
- Set randomly generated password
- Database instance name: capstone
- User name: group3
- database name: db1
- I will send the password by Whatsapp
- We should set up different users, or login through IAM access
- To connect, find endpoint by RDS > Databases > capstone, select tab 'Connectivity & security'
- Using the default setting, connecting to endpoint using Postgres client (psql) did not work,
  neither from home nor from EC-2 instance.

- Creating database using `AWS CLI`:

  ```
  aws rds create-db-instance \
      --db-instance-identifier capstone \
      --db-instance-class t2.micro \
      --engine Postgres \
      --allocated-storage 20 \
      --master-username group3 \
      --master-user-password masteruserpassword \
      --enable-iam-database-authentication 
  ```

### Allow external connections
- I followed the guide at 

  https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_ConnectToPostgreSQLInstance.html

  under 'troubleshooting connections to your SQL instance', 'errors with security group access rules',
  'provide access to your DB instance in your VPC by creating a security group'.

- Created new security group 'DBSecurityGroup', adding an inbound rule for PostgreSQL 
  (allowing TCP to port 5432 from anywhere)
 
- Modified database to add DBSecurityGroup to list of security groups associated with DB instance.

- Select 'apply immediately' and modify database.

- After this change the connection works.  I do not understand why the change was necessary, because the default security group also allows all traffic from everywhere and on all ports.

- Here's how I connected:
  ```
  psql \
  --host=capstone.clihskgj8i7s.us-west-2.rds.amazonaws.com \
  --port=5432 \
  --username=group3 \
  --password \
  --database=db1
  ```

## Connecting through EC2 instance using Jupyter notebook

## Allowing IAM access
- Described at https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.html

- I need to look at this in more detail.

- Creating the following IAM policy? (just as in the example, but changing the resource to our database resource from the 'configuration' tab):

```
  {
    "Version": "2012-10-17",
    "Statement": [
      {
         "Effect": "Allow",
         "Action": [
             "rds-db:connect"
         ],
         "Resource": [
          "arn:aws:rds:us-west-2:270783223265:db:capstone/group3" 
         ]
      }
   ]
}