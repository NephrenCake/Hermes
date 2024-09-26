import os.path

import pyAgrum as gum
import numpy as np
import time


class Bayes_predictor:
    def __init__(self, task_name: str, ):
        self.bn = gum.loadBN(os.path.join(os.path.dirname(__file__), f'{task_name}.bif'))
        self.task_name = task_name
        if task_name == 'got_docmerge':
            self.stages = ["generate1_p", 'generate1_c', 'score1_p', 'score1_c', 'aggregate_p', 'aggregate_c',
                           'score2_p', 'score2_c', 'generate2_p', 'generate2_c', 'score3_p', 'score3_c', ]
            self.sizes = {}
            for key in self.stages:
                self.sizes[key] = 100
        elif task_name == 'factool_code':
            self.stages = ["generate_queries_p", "generate_queries_c", "generate_solutions_p", "generate_solutions_c"]
            self.sizes = {"generate_queries_p": 40, "generate_queries_c": 20, "generate_solutions_p": 40,
                          "generate_solutions_c": 40}
        elif task_name == 'factool_kbqa':
            self.stages = ["extract_claims_p", "extract_claims_c", "generate_queries_p", "generate_queries_c",
                           "verifies_p", "verifies_c"]
            self.sizes = {"extract_claims_p": 40, "extract_claims_c": 20, "generate_queries_p": 10,
                          "generate_queries_c": 10, "verifies_p": 40, "verifies_c": 10}
        elif task_name == 'factool_math':
            self.stages = ["extract_claims_p", "extract_claims_c", "generate_queries_p", "generate_queries_c"]
            self.sizes = {"extract_claims_p": 40, "extract_claims_c": 20, "generate_queries_p": 10,
                          "generate_queries_c": 10, }
        else:
            print('wrong task type !!!')


    def following_predict(self, row, current_stage_id):
        current_stage_id = current_stage_id * 2

        def init_belief(engine, stages, current_stage_id):
            # Initialize evidence
            for var in stages[:current_stage_id]:
                engine.addEvidence(var, 0)

        def update_beliefs(engine, bayesNet, row, stages, current_stage_id):
            # Update beliefs from a given information
            for i, var in enumerate(stages[:current_stage_id]):
                try:
                    choice = [int(i) for i in list(self.bn[var].labels())]
                    if row[i] < np.min(choice):
                        row[i] = np.min(choice)
                    if row[i] > np.max(choice):
                        row[i] = np.max(choice)
                    idx = bayesNet.variable(var).index(str(row[i]))
                    engine.chgEvidence(var, idx)
                except gum.NotFound:
                    # this can happend when value is missing is the test base.
                    pass
            engine.makeInference()

        results = {}
        row = [int(v / self.sizes[self.stages[i]]) for i, v in enumerate(row)]
        ie = gum.LazyPropagation(self.bn)
        init_belief(ie, self.stages, current_stage_id)
        update_beliefs(ie, self.bn, row, self.stages, current_stage_id)

        for target in self.stages[current_stage_id:]:

            ie.addTarget(target)
            marginal = ie.posterior(target)
            results[target] = (np.mean(np.where(marginal.toarray() == np.max(marginal.toarray()))) + np.min(
                [int(i) for i in list(self.bn[target].labels())]) + 0.5) * self.sizes[target]
        return results

    def update(self, row):
        pass


Bayes_predictors = {
    'got_docmerge': Bayes_predictor('got_docmerge'),
    'factool_code': Bayes_predictor('factool_code'),
    'factool_kbqa': Bayes_predictor('factool_kbqa'),
    'factool_math': Bayes_predictor('factool_math'),
}

if __name__ == '__main__':

    # b = Bayes_predictors['got_docmerge']
    # r = b.following_predict([1300, 600],1)
    b = Bayes_predictors['factool_code']
    r = b.following_predict([350, 110], 1)


    print(r)
