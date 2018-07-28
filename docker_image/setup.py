import os
os.system(' echo \'deb http://deb.debian.org/debian jessie-backports main\' \
> /etc/apt/sources.list.d/jessie-backports.list && \
apt update -y && \
apt install --target-release jessie-backports \
      openjdk-8-jre-headless \
      ca-certificates-java \
      --assume-yes &&\
rm -rf /var/lib/apt/lists/* && \
rm -rf /var/cache/oracle-java8-installer')
os.system('echo \'JAVA_HOME= \"/usr/lib/jvm/openjdk-8\"\' >> /etc/environment')

from setuptools import setup

setup(
        name="notapackage",
        version="0.1",
        install_requires=['pyspark', 'numpy']
      )

