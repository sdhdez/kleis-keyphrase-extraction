#!/bin/sh 

dataset=$1
features=$2

function eval_dev {
    # Eval Dev
    devpath="output-dev$features"
    if [ -d $devpath ]
    then
        ls -d1 $devpath/[0-9]*/ | sort -t / -k 2 -n | xargs -I{} sh -c 'echo {}; python ../src/kleis/kleis_data/corpus/semeval2017-task10/eval.py ../src/kleis/kleis_data/corpus/semeval2017-task10/dev/ {} types | grep "avg / total"'
    fi
}

function eval_test {
    #Eval Test
    testpath="output-test$features"
    if [ -d $testpath ]
    then
        ls -d1 $testpath/[0-9]*/ | sort -t / -k 2 -n | xargs -I{} sh -c 'echo {}; python ../src/kleis/kleis_data/corpus/semeval2017-task10/eval.py ../src/kleis/kleis_data/corpus/semeval2017-task10/semeval_articles_test/ {} types | grep "avg / total"'
    fi
}

if [ $dataset = "dev" ]
then 
    eval_dev
elif [ $dataset = "test" ]
then
    eval_test
fi 
