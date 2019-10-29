#!/usr/bin/bash

remote_host=$1

function parse_remote_port()
{
    remote_port=`grep HOST_PORT ~/ide_daemon/ide_daemon.cfg | awk -F '=' '{print $2}'`

    if [[ ${remote_port}"X" == "X" ]];then
        remote_port="22118"
    fi
}

function copyinstalldir()
{
  #echo ${remote_host}
  #echo ${remote_port}
  ret=`IDE-daemon-client --host ${remote_host}:${remote_port} --hostcmd "rm -rf ~/HIAI_PROJECTS/pythonPkg"`
  iRet=`IDE-daemon-client --host ${remote_host}:${remote_port} --hostcmd "mkdir -p ~/HIAI_PROJECTS/pythonPkg"`
  if [[ $? -ne 0 ]];then
    echo "ERROR: create python package error, please check board connection."
    return 1
  fi
  local_path="../install/"
  file_list=`find ${local_path} -name "*"`
  echo ${file_list}
  for local_file in ${file_list}
  do
    echo ${local_path}
    if [[ -d ${file} ]];then
      continue
    fi
    iRet=`IDE-daemon-client --host ${remote_host}:${remote_port} --sync ${local_file} "~/HIAI_PROJECTS/pythonPkg"`
    if [[ $? -ne 0 ]];then
      echo "ERROR: sync ${local_file} failed!"
    fi
  done
  return 0
}

function installPythonPkg()
{
  ret=`IDE-daemon-client --host ${remote_host}:${remote_port} --hostcmd "wc -l ~/HIAI_PROJECTS/pythonPkg/install.sh"`
  if [[ $? -ne 0 ]];then
    echo "ERROR: no pachages to install, exit 1"
    return 1
  fi
  ssh -t HwHiAiUser@${remote_host} "su - root -c \"apt-get update;apt-get upgrade;apt-get install python-dev;cd /home/HwHiAiUser/HIAI_PROJECTS/pythonPkg;bash install.sh\""
  return 0
}

function main()
{
  parse_remote_port

  echo ${remote_port}

  copyinstalldir

  installPythonPkg
}

main

copyinstalldir $@

