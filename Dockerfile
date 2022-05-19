from ubuntu:20.04
MAINTAINER Mariia Ryleeva
RUN apt-get update -y
COPY . /opt/final_project
WORKDIR /opt/final_project
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
