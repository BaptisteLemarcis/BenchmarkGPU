void initDataDistributed(float, float, float*, int);

void FillVec(int, float*, float);

void generateData(int, float*, float*);

void prepareData(float*, float*, int, float*, float*, int, int);

void softmaxError(int, float*, float*, float*);

void softmaxLoss(int, int, float*, float*);

void computeConfusionMatrix(float*, float*, float*, int, int);