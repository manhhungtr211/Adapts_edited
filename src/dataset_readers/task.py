from datasets import load_dataset

class App:
    def __init__(self):
        self.cls_dic = {}

    def add(self, key):
        def adder(cls):
            self.cls_dic[key] = cls
            return cls

        return adder

class Base_Task(object): 
    '''
    base class for loading filled-in input/output from huggingface dataset
    '''
    def __init__(self):
        super().__init__()
        # sum of inference batch size of all the GPUs
        # increase it for faster speed, and decrease it if OOM
        self.inf_bsz = 1

    def get_question(self, entry):
        return entry["input"]
    
    def get_input_strs(self, entry, class_num):
        question = self.get_question(entry)
        return [question] * class_num

    def get_answers(self, entry):
        if 'options' in entry:
            return [' ' + o for o in entry['options']]
        else:
            return [' ' + entry['label']] if isinstance(entry['label'], str) else [' ' + l for l in entry['label']]

    def get_label(self, entry):
        if 'gold_index' in entry:
            return entry['gold_index']
        else:
            return [' ' + entry['label']] if isinstance(entry['label'], str) else [' ' + l for l in entry['label']]

    def get_answer(self, entry):
        if 'gold_index' in entry:
            return ' ' + entry['options'][entry['gold_index']]
        else:
            return ' ' + entry['label'] if isinstance(entry['label'], str) else ' ' + entry['label'][0]


task_map = App()

# ================================ Finance =================================
@task_map.add("FPB")
class FPB(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "weighted_F1"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'FPB', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("FiQA_SA")
class FiQA_SA(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "weighted_F1"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'FiQA_SA', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("Headline")
class Headline(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "Headline"
        self.inf_bsz = 4  # reduced from 32 to avoid OOM on T4 (large activations)

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'Headline', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("NER")
class NER(Base_Task):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "NER"

    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'NER', cache_dir=cache_dir)
        return dataset['test']


@task_map.add("ConvFinQA")
class ConvFinQA(Base_Task):
    def __init__(self):
        super().__init__()
        self.metric = "ConvFinQA"
        self.class_num = 1
        self.inf_bsz = 2  # reduced from 16 to avoid OOM on T4 (long sequences + large activations)
    
    def get_dataset(self, cache_dir=None):
        dataset = load_dataset('AdaptLLM/finance-tasks', 'ConvFinQA', cache_dir=cache_dir)
        return dataset['test']
