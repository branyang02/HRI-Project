#!/bin/bash


trap cleanup SIGINT

cleanup() {
  echo "Shutting down socket connection..."
  sshpass -p '15-003556' ssh double@192.168.1.200 "pkill socat"
  echo "Socket connection shut down."
  exit 0
}

sshpass -p '15-003556' ssh double@192.168.1.200 "echo 'Socket succesfully connects to port ${PORT}!' && socat TCP-LISTEN:${PORT},reuseaddr,fork UNIX-CONNECT:/tmp/doubleapi"

wait $!