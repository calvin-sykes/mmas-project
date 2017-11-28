#!/bin/bash

ssh -XC -L38080:localhost:38080 hrzc75@stargate.dur.ac.uk -t ssh -XC -L38080:localhost:7531 hrzc75@zwicky
