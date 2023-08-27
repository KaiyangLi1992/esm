import pickle
import sys 

sys.path.append("/root/uclasm/") 
sys.path.append("/root/uclasm/rlmodel") 
sys.path.append("/root/uclasm/uclasm/") 
print(sys.path)
from train import run
from model import Model
from config import FLAGS
from saver import saver
from utils import slack_notify, get_ts, OurTimer

with open('toy_dataset.pkl','rb') as f:
    toy_dataset = pickle.load(f)
tm = OurTimer()
train_test_data = [toy_dataset]
model = Model(train_test_data).to(FLAGS.device)
print(model)
saver.log_info('Model created: {}'.format(tm.time_and_clear()))
trained_model = run(train_test_data, saver, model)
