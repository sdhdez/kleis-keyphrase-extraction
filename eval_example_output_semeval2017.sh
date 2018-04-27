# Eval Dev
ls -d1 output-dev/[0-9]*/ | sort -t / -k 2 -n | xargs -I{} sh -c 'echo {}; python corpus/semeval2017-task10/eval.py corpus/semeval2017-task10/dev/ {} types | grep "avg / total"'

#Eval Test
ls -d1 output-test/[0-9]*/ | sort -t / -k 2 -n | xargs -I{} sh -c 'echo {}; python corpus/semeval2017-task10/eval.py corpus/semeval2017-task10/semeval_articles_test/ {} types | grep "avg / total"'
