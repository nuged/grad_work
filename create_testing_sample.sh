#!/bin/bash
DATADIR=mnist
mkdir $DATADIR/small_testing

mkdir $DATADIR/small_testing/0
mkdir $DATADIR/small_testing/1
mkdir $DATADIR/small_testing/2
mkdir $DATADIR/small_testing/3
mkdir $DATADIR/small_testing/4
mkdir $DATADIR/small_testing/5
mkdir $DATADIR/small_testing/6
mkdir $DATADIR/small_testing/7
mkdir $DATADIR/small_testing/8
mkdir $DATADIR/small_testing/9

cp $DATADIR/testing/0/0000??.png $DATADIR/small_testing/0/
cp $DATADIR/testing/1/0000??.png $DATADIR/small_testing/1/
cp $DATADIR/testing/2/0000??.png $DATADIR/small_testing/2/
cp $DATADIR/testing/3/0000??.png $DATADIR/small_testing/3/
cp $DATADIR/testing/4/0000??.png $DATADIR/small_testing/4/
cp $DATADIR/testing/5/0000??.png $DATADIR/small_testing/5/
cp $DATADIR/testing/6/0000??.png $DATADIR/small_testing/6/
cp $DATADIR/testing/7/0000??.png $DATADIR/small_testing/7/
cp $DATADIR/testing/8/0000??.png $DATADIR/small_testing/8/
cp $DATADIR/testing/9/0000??.png $DATADIR/small_testing/9/

#cp $DATADIR/testing/0/0001??.png $DATADIR/small_testing/0/
#cp $DATADIR/testing/1/0001??.png $DATADIR/small_testing/1/
#cp $DATADIR/testing/2/0001??.png $DATADIR/small_testing/2/
#cp $DATADIR/testing/3/0001??.png $DATADIR/small_testing/3/
#cp $DATADIR/testing/4/0001??.png $DATADIR/small_testing/4/
#cp $DATADIR/testing/5/0001??.png $DATADIR/small_testing/5/
#cp $DATADIR/testing/6/0001??.png $DATADIR/small_testing/6/
#cp $DATADIR/testing/7/0001??.png $DATADIR/small_testing/7/
#cp $DATADIR/testing/8/0001??.png $DATADIR/small_testing/8/
#cp $DATADIR/testing/9/0001??.png $DATADIR/small_testing/9/