#!/bin/bash

# Correct patterns
pattern1='MUSICOS, ARTISTAS, EMPRESARIOS Y PRUD ESPECT'
pattern2='DEPORTISTAS, ENTRENADORES, TECNICOS DEPORT'
pattern3='HISTORIADOR, Y PROFESEN CIENCIAS POLITICAS'
pattern4='ESCRITORES, PERIODISTAS Y TRABAJADORES SIMIL'
pattern5='SOCIOLOGO, ANTROPOLOGOS Y ESPECIALISTA SIMIL'
pattern6='BIBLIOTECARIOS, ARCHIVEROS Y ENCARGA DE MUSEO'
pattern7='ESCULTORES, PINTORES, FOTOGRAFOS Y ART SIMI'
pattern8='FILOSOFOS, TRADUCTORES E INTERPRETES'

# Make sure to install csvkit: pip install csvkit
months=(201902 201903 201904 201905 201907 201908 201909 201910 201911
        202001 202002 202003 202004 202005 202007 202008 202009 202010 202011)

for month in ${months[*]}
do
    csvgrep -H -c 1 -m "$month" Dataton_train.csv  > train_$month.csv 
    echo "$month file created."

    sed -i "s/$pattern1/MUSICOS ARTISTAS EMPRESARIOS Y PRUD ESPECT/g" train_$month.csv 
    sed -i "s/$pattern2/DEPORTISTAS ENTRENADORES TECNICOS DEPORT/g" train_$month.csv 
    sed -i "s/$pattern3/HISTORIADOR Y PROFESEN CIENCIAS POLITICAS/g" train_$month.csv 
    sed -i "s/$pattern4/ESCRITORES PERIODISTAS Y TRABAJADORES SIMIL/g" train_$month.csv 
    sed -i "s/$pattern5/SOCIOLOGO ANTROPOLOGOS Y ESPECIALISTA SIMIL/g" train_$month.csv 
    sed -i "s/$pattern6/BIBLIOTECARIOS ARCHIVEROS Y ENCARGA DE MUSEO/g" train_$month.csv 
    sed -i "s/$pattern7/ESCULTORES PINTORES FOTOGRAFOS Y ART SIMI/g" train_$month.csv 
    sed -i "s/$pattern8/FILOSOFOS TRADUCTORES E INTERPRETES/g" train_$month.csv 

    echo "$month cleaned"
done 