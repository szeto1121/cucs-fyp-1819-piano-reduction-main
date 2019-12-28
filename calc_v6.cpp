#pragma GCC optimize("Ofast")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<ctime>
using namespace std;

int n,cnt,oripenalty,sum, t;
int moment[10005][130], minimum[10005][130], original[10005][130];
int larr[20005][6], rarr[20005][6], lhandpos[20005],rhandpos[20005];

int tmppenalty[20005][10];

int tmp[20005][130];

int minpenalty = 2000000000;
int w[6];
int hand_size;
int change = 0;
int changerow[100005], changecol[100005], changedir[100005], changetry[100005];

void copy_score(int from[][130], int to[][130]){
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= 128; j++)
			to[i][j] = from[i][j];
}

int compute_penalty(){
	int sum = 0;
	
	for (int i=1;i<=n;i++)
		for (int j = 0; j <= 6; j++)
			tmppenalty[i][j] = 0;	 
	
	
	for (int i=1;i<=n;i++){
		int min,max;
		int leftcnt = 0;
		
		int cnt = 0;
		for (int j = 1; j <= 128; j++)
			if (moment[i][j] >= 1) cnt++;
		if (cnt > 10) tmppenalty[i][1] += 60;
		
		
		//Left Hand
		min = 999;
		max = 0;
		for (int j = 1; j <= 60; j++)
			if (moment[i][j] >= 1){
				if (min == 999) min = j;
				larr[i][++leftcnt] = j;
			}
		max = larr[i][leftcnt];
		if ((min == 999) && (max == 0)) 
			lhandpos[i] = lhandpos[i-1];
		else if (abs(lhandpos[i-1] - (min+max)/2) >=6)
			lhandpos[i] = (min+max)/2;
		else lhandpos[i] = lhandpos[i-1];
		
		//Right Hand
		int rightcnt= 0;
		min = 999;
		max = 0;
		for (int j = 61; j <= 128; j++){
			if (moment[i][j] >= 1){
				if (min == 999) min = j;
				rarr[i][++rightcnt] = j;
			}
		}
		max = rarr[i][rightcnt];
		if ((min == 999) && (max == 0))
			rhandpos[i] = rhandpos[i-1];
		else if (abs(rhandpos[i-1] - (min+max)/2) >=6)
			rhandpos[i] = (min+max)/2;
		else rhandpos[i] = rhandpos[i-1];
	
		
		//gap
		int gap_lim[5] = {0, 5, 4, 4, 6};
		int gap_pen[5] = {5, 15, 5, 5};
		for (int j = 1; j <= 4; j++){
			if (larr[i][j] > 0 && larr[i][j + 1] > 0 && larr[i][j + 1] - larr[i][j] >= gap_lim[j])
				tmppenalty[i][2] += gap_pen[j];
			if (rarr[i][j] > 0 && rarr[i][j + 1] > 0 && rarr[i][j + 1] - rarr[i][j] >= gap_lim[j])
				tmppenalty[i][2] += gap_pen[j];
		}
			
		//crowdness ~ busyness 
		if (leftcnt == 4)  tmppenalty[i][3] += 50;
		else if (leftcnt == 5) tmppenalty[i][3] += 100;
		if (rightcnt == 4)  tmppenalty[i][3] += 50;
		else if (rightcnt == 5) tmppenalty[i][3] += 100;
			
		//range
		if ((larr[i][leftcnt] - larr[i][1]) > hand_size) tmppenalty[i][4] += 200;
		else if ((larr[i][leftcnt] - larr[i][1]) > hand_size+2) tmppenalty[i][4] += 400;
		
		if ((rarr[i][rightcnt] - rarr[i][1]) > hand_size) tmppenalty[i][4] += 200;
		else if ((rarr[i][rightcnt] - rarr[i][1]) > hand_size+2) tmppenalty[i][4] += 400;
		
		
		//hand movement busyness
		if (i>=2){
			//if (lhandpos[i]-lhandpos[i-1]>=5)
			//	penalty[i]+=10 * abs(lhandpos[i]-lhandpos[i-1])*4 ;
			//else if (lhandpos[i]-lhandpos[i-1]>=10)
			//	penalty[i]+=30;
			//else if (lhandpos[i]-lhandpos[i-1]>=15)
			//		penalty[i]+=50;
			tmppenalty[i][5] += abs(lhandpos[i]-lhandpos[i-1])*10;
			tmppenalty[i][5] += abs(rhandpos[i]-rhandpos[i-1])*10;
			
			/*if (rhandpos[i]-rhandpos[i-1]>=5)
				penalty[i]+=10;
			else if (rhandpos[i]-rhandpos[i-1]>=10)
				penalty[i]+=30;
			else if (rhandpos[i]-rhandpos[i-1]>=15)
				penalty[i]+=50;*/
		}
		
		//difference
		int diff_count = 0;
		for (int j = 1; j <= 128; j++)
			if ((!!moment[i][j]) != (!!original[i][j]))
				diff_count++;
		tmppenalty[i][6] = diff_count * 10;
		
		for (int j = 1; j <= 6; j++){
			tmppenalty[i][0] += tmppenalty[i][j]*w[j];
			sum += tmppenalty[i][j]*w[j];
		}
		//sum += penalty1[i]*w1 + penalty2[i]*w2 + penalty3[i]*w3;
	} 
	return sum;
}

int main(int argc, char* argv[]){
	FILE *fp;
	if (argc >= 2)
		fp = fopen(argv[1],"r");
	else
		fp = fopen("sample.txt", "r");
	 
	fscanf(fp, "%d",&n);
	
	if (argc >=7){
		
		
		hand_size = atoi(argv[2]);
		w[1] = atoi(argv[3]);
		w[2] = atoi(argv[4]);
		w[3] = atoi(argv[5]);
		w[4] = atoi(argv[6]);
	}
	else {
		hand_size = 12;
		w[1] = 1; w[2]=1; w[3]=1; w[4]=1;
	}
	
	
	//input 
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= 128; j++)
			fscanf(fp, "%d", &original[i][j]);
	fclose(fp);
	
	copy_score(original, moment);
	int nowpenalty = compute_penalty();
    minpenalty = nowpenalty;
	oripenalty = nowpenalty;
	copy_score(moment, minimum);
	
	printf("final penalty = %d\n", minpenalty);
	
	for (int i=0;i<=n/10;i++){
		printf("%3d              ",i); 
		for (int j=1;j<=10;j++){
			printf("%4d ", tmppenalty[i*10+j][0]);
			//printf("%4d ",penalty1[i*10+j]*w1 + penalty2[i*10+j]*w2 + penalty3[i*10+j]*w3);
		}
		printf("\n");
	}
	
	system("pause");
	
	
	//TRY to modify some notes
	 
	srand(time(0));
	int r,c,choice;
	int k = 0;
	int con = 0; //# of consective no use changes
	while (con < 10000){
		k++;
		
		copy_score(minimum, moment);
		do{
			r = (rand() % (n - 1)) + 2;
			int have = 0;
			for (int i = 1; i <= 128; i++)
				if (moment[r][i] > 0)
					have = 1;
			if (!have) continue;
			do{
				c = (rand() % 128) + 1;
			} while (moment[r][c] == 0);
		}while (0);
		
		choice = (rand() % 2);
		// printf("%4d %3d %3d ",r,c,choice);fflush(stdout);
		//system("pause"); 
		if (choice == 0){
			moment[r][c]--;
			moment[r][c-12]++;
		}
		else{
			moment[r][c]--;
			moment[r][c+12]++;
		}
	
		int nowpenalty = compute_penalty();
		
		
		
		if (nowpenalty < minpenalty){
			printf("TRY #%d     final penalty = %d\n", k, nowpenalty);	
			copy_score(moment, minimum);
			t = k;
			con = 0;
			change++;
			changetry[change] = k;
			changerow[change] = r;
			changecol[change] = c;
			changedir[change] = choice;
			minpenalty = nowpenalty;
		}	
		else 
			con++;
	}
	
	
	printf("Original Penalty = %d\n",oripenalty);
	printf("MINIMUM PENALTY = %d    TRY %d \n",minpenalty,t);
	printf("Total number of changes = %d\n", change);

	for (int i=1;i<=change;i++)
		printf("Try = %d %d %d %d\n", changetry[i], changerow[i], changecol[i], changedir[i]);
	
	fp = fopen("after.txt","w");
	for (int i=1;i<=n;i++){
		for (int j=1;j<128;j++)
			fprintf(fp, "%d ", !!minimum[i][j]);
		fprintf(fp, "%d\n",!!minimum[i][128]);
	}

	return 0;

}
