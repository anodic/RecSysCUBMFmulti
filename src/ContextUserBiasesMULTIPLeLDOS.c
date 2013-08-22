/*
 ============================================================================
 Name        : NewMF.c
 Author      : Ante Odic
 Version     :
 Copyright   : Your copyright notice
 Description : 	Performs matrix factorization with context pre-filtering on LDOS database.
				Evaluation measure is RMSE.
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nrutil.c"
#include <string.h>


// global variables
int *usageHistory_userID;
int *usageHistory_itemID;
int **context_training;
int **context_testing;
unsigned char *usageHistory_score;

float globalBias, **userPerContextBias, *itemBias, *userBias;
float **pUF, **pIF;
float contextMultipleBiases[13][258][8];

//dataset parameters
int numOfUsageHistory 		= 1491;//full:1951;//cleaned:1189;//trainNoLast:1464;//1518;//1345;//1492;//1282;//1429;
int numOfFeatures 			= 10;
int numOfEpochs				= 100;
int numOfUsers				= 267;//cleaned:257;//trainNoLAst:200;
int numOfItems				= 4323;//full:4323;//cleaned:4306;//trainNoLast:4138;
int testSetSize 			= 119;//full:199;//cleaned:109;//trainNoLast:147;//1492;//147;//1429;
int numOfContextVariables 	= 12;
int numOfContextClassesALL[12] = {4,3,4,3,5,7,7,7,3,2,2,2};
int numOfContextClasses =0;
int contextOfInterest = 0 ;


//test set
//char *validationSetFileName = "D:/00xBeds/03-MatrixFactorizationWithContext/data/LDOScontextDB/LDOScontextTEST.txt";
char *validationSetFileName = "D:/00xBeds/03-MatrixFactorizationWithContext/data/LDOScontextDB/LDOScomoda-Original-Folding/folds/LDOScomodaTestOriginal1.txt";
//train set
//char *filename = "D:/00xBeds/03-MatrixFactorizationWithContext/data/LDOScontextDB/LDOScontextTRAINnoLAST.txt";
char *filename = "D:/00xBeds/03-MatrixFactorizationWithContext/data/LDOScontextDB/LDOScomoda-Original-Folding/folds/LDOScomodaTrainOriginal1.txt";



//matrix factorization parameters
float initValue 		= 0.03;
float learningRate		= 0.00001;//0.001 ---->0,00001
float lRateItemBias		= 0.0001;//----> 0,0001
float lRateUserBias		= 0.00001;//---->0,00001
float K					= 0.005;//0.02 ---->0,005

int loadUsageHistory(char *fileName);
int calculateStaticBiases();
int startTraining();
int startValidating();
int getNextUserValidationData(FILE *fp,int **testBlock, int itemNumber);
int getContextualInformation();

int main(void) {

	int c, first, second, third;
	int usrCnt;
	int conClsCnt;
	fflush(stdout);

// initialize contextMultipleBiases array
	for (first = 0; first <= 12; first++){ 				//first = number of contextual information
		for (second = 0; second <= 200; second++){ 		//second = number of users
			for (third = 0; third <= 7; third++){		//third = number of max context values
				contextMultipleBiases[first][second][third] = 0;
			}
		}
	}

	// initialize buffers
	usageHistory_itemID 	= ivector(1,numOfUsageHistory);
	usageHistory_score 		= cvector(1,numOfUsageHistory);
	usageHistory_userID 	= ivector(1,numOfUsageHistory);

	// initialize feature vectors
	pUF = matrix(0,numOfUsers,0,numOfFeatures);
	pIF = matrix(0,numOfItems,0,numOfFeatures);

	// initialize bias vectors
	itemBias = vector(0,numOfItems);
	userBias = vector(0,numOfUsers);



	//execute this sequence
	loadUsageHistory(filename);
	getContextualInformation();


	for (c=1;c<=numOfContextVariables;c++)
	{
	//c=7;
		contextOfInterest =c;
	numOfContextClasses =  numOfContextClassesALL[contextOfInterest-1];

	//printf("odabrani kontekst je %d\n",numOfContextClassesALL[contextOfInterest-1]); fflush(stdout);

	calculateStaticBiases();

	for (usrCnt = 1; usrCnt <=200; usrCnt++)
		{
			for (conClsCnt = 1; conClsCnt <= numOfContextClasses; conClsCnt++)
			{
				contextMultipleBiases[c][usrCnt][conClsCnt] = userPerContextBias[usrCnt][conClsCnt] - userBias[usrCnt];
			}
		}
	}



	startTraining();
	startValidating();



	printf("end of app\n."); fflush(stdout);

	return EXIT_SUCCESS;
}


/**
 * Loads contextual information from training and testing set into memory
 */

int getContextualInformation(){
	char line[200],*ret;
	int i, temp1, temp2, temp3;

	FILE *trainingFp = fopen(filename,"r");
	if (trainingFp == NULL) perror ("Error opening file");
	FILE *testingFp = fopen(validationSetFileName,"r");
	if (testingFp == NULL) perror ("Error opening file");


	//printf("usao u getContextual\n"); fflush(stdout);

	context_training = imatrix(1,numOfUsageHistory,1,numOfContextVariables);
	context_testing = imatrix(1,testSetSize,1,numOfContextVariables);

	//printf("napravio matrice\n"); fflush(stdout);
	//get context from training set
	for(i=1;i<=numOfUsageHistory;i++){
		ret = fgets(line , 200, trainingFp);
		//printf("uzeo %d liniju\n", i); fflush(stdout);
		sscanf(line, "%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d",&temp1,&temp2,&temp3,&context_training[i][1],&context_training[i][2],&context_training[i][3],&context_training[i][4],&context_training[i][5],&context_training[i][6],&context_training[i][7],&context_training[i][8],&context_training[i][9],&context_training[i][10],&context_training[i][11],&context_training[i][12]);
	}

	//get context from testing set
	for(i=1;i<=testSetSize;i++){
		ret = fgets(line , 200, testingFp);
		//printf("uzeo %d testing liniju\n", i); fflush(stdout);
		sscanf(line, "%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d",&temp1,&temp2,&temp3,&context_testing[i][1],&context_testing[i][2],&context_testing[i][3],&context_testing[i][4],&context_testing[i][5],&context_testing[i][6],&context_testing[i][7],&context_testing[i][8],&context_testing[i][9],&context_testing[i][10],&context_testing[i][11],&context_testing[i][12]);
	}

	return 1;
}






/**
 * Loads the whole usage history data set in memory
 */
int loadUsageHistory(char *fileName){
	FILE *fp = fopen(fileName,"r");
	if (fp == NULL) perror ("Error opening file");
	char line[200];
	int i,t1,t2;
	int userID,itemID,score;

	t1 = clock();
	printf("Start loading  usage history ...\n");

		for (i=1;i<=numOfUsageHistory;i++){
			fgets(line,200,fp);
			sscanf(line,"%d;%d;%d",&userID,&itemID,&score);
			usageHistory_userID[i] = userID;
			usageHistory_itemID[i] = itemID;
			//usageHistory_userID[i] = (unsigned char) userID;
			//usageHistory_itemID[i] = (unsigned char) itemID;
			usageHistory_score[i] = (unsigned char) score;
			//usageHistory_time[i] = time;


			////////////////// test loading////////////////
			//printf("\n user %d item %d rating %d\n",usageHistory_userID[i], usageHistory_itemID[i], (int)usageHistory_score[i] );
			///////////////////////////////////////////////


			if (i==numOfUsageHistory) {
				printf("Finished %d\n",i);
				fflush(stdout);
				t2 = clock();
				printf("done in %f secs\n",(float)(t2-t1)/1000); fflush(stdout);
				return -1;
			}
			if (fmod((double)i,1000000.0) == 0){
			printf("%d\n",i); fflush(stdout);}
		}

	fclose(fp);
	t2 = clock();
	printf("done reading usage history in %f secs\n",(float)(t2-t1)/1000); fflush(stdout);



	return 1;
}

/**
 * calculates the static biases from the training set
 */
int calculateStaticBiases(){
	int userID, itemID, i, j;

	int score, index, contextualInformation;
	int **countOfUsersPerContext = imatrix(0,numOfUsers,0,numOfContextClasses);
	int *countOfItems = ivector(0,numOfItems);
	int *countOfUsers = ivector(0,numOfUsers);

	userPerContextBias = matrix(0,numOfUsers,0,numOfContextClasses+1);

	printf("Start calculating static biases...\n"); fflush(stdout);
	for (i=0; i < numOfItems+1; i++){
		countOfItems[i] = 0;
	}

	for (i=0; i < numOfUsers+1; i++){
		countOfUsers[i] = 0;
		}


	for (i=0; i < numOfUsers+1; i++){
			for(j=0;j<=numOfContextClasses;j++){
				countOfUsersPerContext[i][j] = 0;
				userPerContextBias[i][j]=0;
			}
	}

	index = 0;
	for(j=1;j<=numOfUsageHistory;j++){
		// numOfLines = getNextUserTrainingData(fp,userHistory,maxUserHistory);
		userID 	= usageHistory_userID[j];
		itemID 	= usageHistory_itemID[j];
		score 	= usageHistory_score[j];
		contextualInformation 	= context_training[j][contextOfInterest];
		if(contextualInformation==0)
					continue;

			if (index==0){
				globalBias = score;
			}

			else{

				float N = (float) index;
				globalBias = globalBias *(N)/(N+1) + score/(N+1);
			}
			index++;


			if (countOfItems[itemID] == 0){
				itemBias[itemID] = score;
			}
			else{
				float N = (float)countOfItems[itemID];
				itemBias[itemID] = itemBias[itemID] *(N)/(N+1) + score/(N+1);
			}

			if (countOfUsers[userID] == 0){
				userBias[userID] = score;
			}
			else{
				float N = (float)countOfUsers[userID];
				userBias[userID] = (N/(N+1) * userBias[userID] )+ score/(N+1);
			}


			if (countOfUsersPerContext[userID][contextualInformation] == 0){
					userPerContextBias[userID][contextualInformation] = score;
			}
			else{
				float N = (float)countOfUsersPerContext[userID][contextualInformation];
				userPerContextBias[userID][contextualInformation] = (N/(N+1) * userPerContextBias[userID][contextualInformation] )+ score/(N+1);
			}


			countOfItems[itemID]++;
			countOfUsers[userID]++;
			countOfUsersPerContext[userID][contextualInformation]++;

		}	// end of lines


	//printf("od prvog: za item %d ::: global bias: %f, user21 bias: %f user1 context1 bias: %f, item bias: %f\n",itemID, globalBias, userBias[21],userPerContextBias[21][1],  itemBias[3939] ); fflush(stdout);

	//printf("izracunao biase"); fflush(stdout);
	for (i=0;i<= numOfItems;i++){
		if(itemBias[i]!=0)
			itemBias[i]-=globalBias;
	}
	for (i=0;i<=numOfUsers;i++){
		userBias[i]-=globalBias;
	}

	for (i=1;i<=numOfUsers;i++){
		for(j=0;j<=numOfContextClasses+1;j++){
			if(userPerContextBias[i][j]!=0)//
				userPerContextBias[i][j]-=globalBias;
			else//
				userPerContextBias[i][j]=userBias[i];//
		}
	}



	printf("Ended calculating static biases...\n");

	return 1;
}


/**
 * Calculates the predicted score with static biases
 */
inline float predictScoreWithStaticBias(int userID, int itemID, int numOfFeatures, int *contextualInformation, int what){
	int f;
	float estimatedScore;
	float sumOfContextBiases=0;
	int conInd;

	estimatedScore = globalBias + userBias[userID] + itemBias[itemID];


	for(conInd = 1; conInd <=numOfContextVariables; conInd++)
	{
		//if (userID == 70 && itemID ==3) printf("userID: %d; itemID: %d; na pocetku - sumOfContextBiases = %f \n", userID, itemID, sumOfContextBiases); fflush(stdout);
			if (contextualInformation[conInd]==0){
			sumOfContextBiases = sumOfContextBiases + 0;
			//if(userID == 70 && itemID ==3) printf("kontext je NULA, sumOfContextBiases = %f \n", sumOfContextBiases); fflush(stdout);
		}
			//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//nolast//else if (conInd==2 ||conInd==4 || conInd==7 || conInd==8 || conInd==9 || conInd==10 || conInd==11 || conInd==12) //bad
		//nolast//else if (conInd == 1 ||conInd == 3 ||conInd == 5 ||conInd == 6) // good
		//nolast//else if ( conInd > 300) // all

		else if (conInd==1 ||conInd==2 || conInd==6 || conInd==7 || conInd==8 || conInd==9 || conInd==10 || conInd==12) //bad
		//else if (conInd == 3 ||conInd == 4 ||conInd == 5 || conInd==11) // good
		//else if ( conInd > 300) // all
		{
			sumOfContextBiases = sumOfContextBiases + 0;
			//if (userID == 70 && itemID ==3) printf("kontext je jedan, sumOfContextBiases = %f \n", sumOfContextBiases); fflush(stdout);
		}
		else
		{
			sumOfContextBiases = sumOfContextBiases + contextMultipleBiases[conInd][userID][(contextualInformation[conInd])];
			//if (userID == 70 && itemID ==3) printf("kontext nije jedan, sumOfContextBiases = %f \n", sumOfContextBiases); fflush(stdout);
		}
	}
	estimatedScore = estimatedScore+ sumOfContextBiases;
	for(f=0;f<numOfFeatures;f++)
	{
		estimatedScore += pIF[itemID][f]*pUF[userID][f];
	}

	if (estimatedScore<1){
		estimatedScore=1;
	}else if (estimatedScore>5){
		estimatedScore=5;
	}

	return estimatedScore;
}



/**
 * Training features from the training set
 */
int startTraining(){

	int i,f,t1,t2,e;
	int userID, itemID, trueScore, contextualInformation[13];
	float predictedScore, err, tempUF, tempIF;
	int conInd;

	////////////// test users, items, ratings////////////////
	//for (i=1;i<=4;i++)
	//printf("\n user %d item %d rating %d\n",usageHistory_userID[i], usageHistory_itemID[i], (int)usageHistory_score[i] );
	///////////////////////////////////////////////



	t1 = clock();
	printf("Start initializing features ...\n"); fflush(stdout);
	for (f=0;f<numOfFeatures;f++){
		for (i=0;i<numOfItems;i++){
			pIF[i][f] = initValue;
		}
		for (i=0;i<numOfUsers;i++){
			pUF[i][f] = initValue;
		}
	}
	t2 = clock();
	printf("done in %f secs\n",(float)(t2-t1)/1000); fflush(stdout);

	t1 = clock();
	printf("Start training features ...\n"); fflush(stdout);
	for (f=0;f<numOfFeatures;f++){
		//printf("Start feature %d ...\n",f); fflush(stdout);
		for (e=0;e<numOfEpochs;e++){
			//printf("Start epoch %d ...\n",e); fflush(stdout);

			//learningRate = learningRates[e];
			for (i=1;i<=numOfUsageHistory;i++){
				//printf("i je %d ...\n",i); fflush(stdout);


				userID = (int) usageHistory_userID[i];
				//printf("userID: %d\n",userID);
				itemID = (int) usageHistory_itemID[i];
				trueScore = (int) usageHistory_score[i];
				for (conInd = 1; conInd<=numOfContextVariables; conInd++){
					contextualInformation[conInd] = context_training[i][conInd];
				}

				//if(contextualInformation==-1)
					//		continue;

				//if(f>30)
				//	printf ("Item ID = %d UserID = %d context= %d\n", itemID, userID, contextualInformation); fflush(stdout);

				predictedScore = predictScoreWithStaticBias	(userID,itemID,f+1,contextualInformation, 0);

				err = ((float) trueScore - predictedScore);

				tempUF = pUF[userID][f];
				tempIF = pIF[itemID][f];



				//pUF[contextualInformation][userID][f] = tempUF + err * tempIF * learningRate;
				//****************************************
				//teach 0 every time, even in the case of a different context
				//if (contextualInformation!=0)
				//	pUF[0][userID][f] = tempUF + (err * tempIF-K*tempUF )* learningRate;

				//if(userID == 28 && (itemID==19 || itemID ==33))
				//										printf ("newPUF = %f\n", pUF[userID][f] ); fflush(stdout);
				//pIF[itemID][f] = tempIF + err * tempUF * learningRate;
				//if(userID == 28 && (itemID==19 || itemID ==33))
				//										printf ("newPIF = %f\n", pIF[itemID][f] ); fflush(stdout);

				//if(itemID ==215)
								//	fprintf (fp,"nakon-> user: %d; item: %d; pscore: %f; lRate= %f; err= %f; npUF= %f; npIF= %f\n",userID, itemID, predictedScore, learningRate, err,  pUF[userID][f], pIF[itemID][f]); fflush(stdout);


				//regularisation
				pUF[userID][f] = tempUF + (err * tempIF - K*tempUF) * learningRate;
				pIF[itemID][f] = tempIF + (err * tempUF - K*tempIF) * learningRate;

				//printf("%d",6);fflush(stdout);
				 //test
				//if (itemID==22757)
				//{
				//	printf("\n item 22757 features %f %f \n", pIF[itemID][0], pIF[itemID][1]);
				//}
				//if (userID==0)
				//{
				//	printf("\n user 0 features %f %f \n", pUF[userID][0], pUF[userID][1]);
				//}

			}
		}

	}
	t2 = clock();
	printf("done in %f secs\n",(float)(t2-t1)/1000); fflush(stdout);
	printf("Time per one feature-one epoch =  %f secs\n",(float)(t2-t1)/(1000*numOfEpochs*numOfFeatures)); fflush(stdout);

	/*
	int j;
	for(j=0;j<20;j++){
		printf("\n user features: ");
		for (i=0;i<numOfFeatures;i++){
		printf(" %f ", pUF[j][i]);
	}
	}
	for(j=0;j<20;j++){
	printf("\n item features: ");
		for (i=0;i<numOfFeatures;i++){
			printf(" %f ", pIF[j][i]);
		}
	}
	*/

	//fclose(fp);
	return 1;
}


/**
 * Performs the validation of the previously trained features on the validation dataset
 */
int startValidating(){
	int i,inspect, conInd;
	float eSc;
	printf("Start validating ...\n"); fflush(stdout);

	FILE *fp;
	float RMSE = 0, trueScore;
	int itemNumber = 0, userID, itemID, contextualInformation[13];
	int testBlockSize = 147;
	float *estimatedScores 	= vector(0,testSetSize);
	float *scoreDifferences	= vector(0,testSetSize);
	int **testBlock = imatrix(0,testBlockSize,0,4);


	//char *validationSetFileName = "D:/00xBeds/03-MatrixFactorizationWithContext/data/LDOScontextDB/testSetOURdatabaseCLEAN.txt";
	//char *validationSetFileName = "D:/WORK/contextRecommenderProject/camra2011/public_eval_t1CLEAN.txt";




	fp = fopen(validationSetFileName,"r");
	while( (inspect = getNextUserValidationData(fp,testBlock, itemNumber)) > 0 && itemNumber<testSetSize){
		for (i=0;i<testBlockSize;i++){
			userID		= testBlock[i][0];
			itemID		= testBlock[i][1];
			trueScore 	= testBlock[i][2];


			for (conInd = 1; conInd<=numOfContextVariables; conInd++){
				//printf("\n for petlja"); fflush(stdout);
				//printf("\n in validation...  i = %d ; conInd = %d; context in testing = %d",i, conInd, context_testing[i][conInd]); fflush(stdout);
				contextualInformation[conInd] = context_testing[i+1][conInd];
				//printf("\n in validation... contextualInformation = %d", contextualInformation[conInd]); fflush(stdout);
			}


			//if(i==8){
			//	printf("validiram 9. usr: %d,  item: %d,  score: %f,  context: %d  \n",userID,itemID,trueScore,contextualInformation); fflush(stdout);
			//}

			// without static biases
			//eSc = predictScore(userID,itemID,numOfFeatures);
			// with static biases
			eSc = predictScoreWithStaticBias(userID,itemID,numOfFeatures, contextualInformation,1);
			//eSc = 50;

		   // printf("izracunao \n"); fflush(stdout);


			estimatedScores[itemNumber] = eSc;
			/*if (i==34){
				printf("user: %d   ", userID); fflush(stdout);
				printf("item: %d   ", itemID); fflush(stdout);
				printf("line: %d   ", itemNumber+1); fflush(stdout);
				printf("%f --> %f\n",trueScore,eSc); fflush(stdout);
			}*/
			scoreDifferences[itemNumber] = trueScore - eSc;
			itemNumber++;
			if(itemNumber == testSetSize)
				break;
		}
	}
	fclose(fp);
	for (i=0;i<itemNumber;i++){
		RMSE += scoreDifferences[i]*scoreDifferences[i];
	}
	RMSE = sqrtf(RMSE/itemNumber);
	printf("For context %d, final RMSE = %f\n",contextOfInterest,RMSE);

	return EXIT_SUCCESS;
}


/**
 * Retrieves the next chunk of lines from the validation dataset file
 */
int getNextUserValidationData(FILE *fp,int **testBlock, int itemNumber){
	char line[200],*ret;
	int i, userID, itemID, score;
	int testBlockSize = 147;

	for(i=0;i<testBlockSize;i++){

		ret = fgets(line , 200,fp);
		sscanf(line, "%d;%d;%d",&userID,&itemID,&score);
		if ((itemNumber+i) > testSetSize) break;
		testBlock[i][0] = userID;
		testBlock[i][1] = itemID;
		testBlock[i][2] = score;

		/*if(i==8){
						printf("uzimam 9. usr: %d,  item: %d,  score: %d \n",userID,itemID,testBlock[i][2]); fflush(stdout);
					}*/


		// todo: add time and date when needed
	}
	return 1;
}
