# bio-eureka
BioEureka is a space biology knowledge engine that boosts discovery. This platform is able to fetch space biology publications from PMC, summarize them, cluster them into topics and find connections between them, using knowledge graphs.
It is accessible and deployed in Railway https://bioeureka-production.up.railway.app/, but it is possible to run the web application locally.
## Installation
To clone the repository to your computer usen the following command line:
```bash
  git clone https://github.com/almylonas/bio-eureka
```
Install all the required libraries, using the command:
```bash
  cd bio-eureka
  pip install -r requirements.txt
```
Then, if you want to run the web app locally, use the command:
```bash
  python app.py
```
The app should be accessible through http://localhost:5000/