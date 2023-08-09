# Codes for Paper "Cache-Enabled Federated Learning Systems"
Codes of algorithm implementations and experiments for paper:
> Yuezhou Liu, Lili Su, Carlee Joe-Wong, Stratis Ioannidis, Edmund Yeh, and Marie Siew, "Cache-Enabled Federated Learning Systems", In The Twenty-fourth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc’23), October 23–26, 2023, Washington, DC, USA. ACM, New York, NY, USA

Please cite this paper if you intend to use this code for your research.

## Usage
* [``logistic_regression.py``](logistic_regression.py): experiments for binary logistic regression
* [``multi_logit.py``](multi_logit.py): experiments for multilogit
* [``MLP_FL.py``](MLP_FL.py): experiments for MLP
* [``CNN_FL.py``](CNN_FL.py): experiments for CNN
* [``CNN_FL_sel.py``](CNN_FL_sel.py): experiments for CNN (with single FL algorithm)
* [``CNN_FL_sampling.py``](CNN_FL_sampling.py): experiments for CNN with client sampling
* [``multi_logit_asynch.py``](multi_logit_asynch.py): experiments for asynchronous FL
* [``read_mobiperf.py``](read_mobiperf.py): read communication and computation delays from real traces

## Data Traces
* [``Measurement1``](Measurement1), [``Measurement2``](Measurement2), [``Measurement3``](Measurement3), [``Measurement4``](Measurement4), and [``Measurement5``](Measurement5) are real communication traces collected by Mobiperf (https://www.measurementlab.net/tests/mobiperf/)
* [``Failure%20Models.xlsx``](Failure%20Models.xlsx): computation delay trace of clients doing FL