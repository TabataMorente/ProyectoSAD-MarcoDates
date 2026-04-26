from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_classic.evaluation import ExactMatchStringEvaluator
from datasets import load_dataset
import argparse

#run "ollama pull gemma2:2b" in your terminal before running this script

parser=argparse.ArgumentParser(description='casiMedicos ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='validation', help='split')
parser.add_argument('--sample', type=int, default=-1, help='sample')
args=parser.parse_args()

template = """Respond to the realworld medical entrance exam question with a single letter: A,B,C,D or E without any explanation.
Question: {question}
Answer:{answer}"""
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model,temperature=0,num_predict=1,top_k=10,top_p=0.5) #deterministic
chain = prompt | model

evaluator = ExactMatchStringEvaluator()
ok = 0
wrongOut = 0

dataset="HiTZ/casimedicos-exp"
casimed = load_dataset(dataset, args.lang) #check huggingface datasets for details
c2l=['A','B','C','D','E']
for n,instance in enumerate(casimed[args.split]):
    if n==args.sample: break #speed up things use only the first n instances
    qa=(instance['full_question']+"\n"
            +" A: "+instance['options']['1']+"\n"
            +" B: "+instance['options']['2']+"\n"
            +" C: "+instance['options']['3']+"\n"
            +" D: "+instance['options']['4']+"\n"
            +" E: "+str(instance['options'].get('5', '')))
    ans=chain.invoke({'question': qa,'answer': ''}).strip() #remove newLine
    if ans not in c2l: wrongOut+=1
    score=evaluator.evaluate_strings(prediction=ans,reference=c2l[instance['correct_option']-1])['score']
    if score==1.0: ok+=1
    acc=round(100*ok/(n+1),2)
    print("| "+args.model+" | "+dataset+"-"+args.lang+"-"+args.split+" | n: "+str(n+1)+" | acc: "+str(acc)+" | inc: "+str(wrongOut)+" |")