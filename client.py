import os
import yaml
import time
import torch
import psutil



class Client:
    def __init__(self, logger, cid, device, model_class, model_args, data_path, dataset_id, train_batch_size, test_batch_size, minibatch_time):
        self.cid = cid
        self.device = device
        self.logger = logger
        self.minibatch_time = minibatch_time
        
        self.model = model_class(cid, model_args)
            
        
        with open(
            os.path.join(
                data_path, f"part_{cid}", dataset_id, "train_dataset_config.yaml"
            ),
            "r",
        ) as m:
            meta = yaml.safe_load(m)
            self.num_items = meta["metadata"]["num_items"]
            self.data_distrb = meta["metadata"]["label_distribution"]
    
        
        
        
       # Load data directly into loaders (no need to wrap again)
        self.train_loader, self.test_loader = self.model.load_data(logger, data_path, dataset_id, cid, train_batch_size, test_batch_size)

        # Extract labels from dataset inside the loader
        try:
            dataset = self.train_loader.dataset
            self.train_labels = [dataset[i][1] for i in range(len(dataset))]
        except Exception as e:
            print(f"❌ Client {cid}: Could not extract train_labels due to: {e}")
            self.train_labels = []

        # Estimate round time based on dataset length
        self.roundtime = self.minibatch_time * len(self.train_loader.dataset)



        
  
        self.train_metrics = dict()
        self.test_metrics = dict()
        
        self.time_util = dict()


        
        
    def get_num_model_params(self):
        self.logger.info(f"NUM_MODEL_PARAMS: {self.model.count_parameters()}")
        
    def expected_time_util(self):
        return self.pareto[self.power_mode][1]
    
    def update_util(self,round_id ,epochs):
        self.time_util[round_id] = self.roundtime*epochs
        
    """ def train(self, round_id, args):
        self.train_metrics[round_id] = self.model.train_model(self.logger, self.train_data, args, self.device)
        self.update_util(round_id, args["epochs"]) """

    def train(self, round_id, args):
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=args["lr"])
        loss_fn = torch.nn.CrossEntropyLoss()

        batch_times = []

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            start = time.time()

            optimizer.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            end = time.time()
            batch_times.append(end - start)

            if batch_idx >= 100:
                break

        avg_time = sum(batch_times) / len(batch_times)
        print(f"[Client {self.cid}] Avg. minibatch time: {avg_time:.4f} seconds")
        # Memory profiling (optional but useful)
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        print(f"[Client {self.cid}] Memory usage: {memory_mb:.2f} MB")


        self.update_util(round_id, args["epochs"])





        
    def test(self, round_id):
        self.test_metrics[round_id] = self.model.test_model_client(self.logger, self.test_data,self.cid)
        self.last_round_tested = round_id
    
    def test_global_data(self,round_id,data):
        self.test_metrics[round_id] = self.model.test_model_client(self.logger, data,self.cid)
        self.last_round_tested = round_id
        
        