import math
import os
import numpy as np

class Postprocessor:
    def __init__(self, algo):
        """Instantiate a postprocessor using external files
        
        The format of the input file for the external program should be as follows:
            The first line contains an integer n, the number of moments of the score
            For the following n lines, each line consists of 128 integers, where the 
            i-th  value is 1 if the note with pitch i is played at that moment, and 0 
            if otherwise.
            
        The format of the output file for the external program is similar to the input file:
            There are n lines in the output denoting the post-processed version of the score. 
            Each line consists of 128 integers, and the i-th  value is 1 if the note with pitch 
            i is played at that moment, and 0 if otherwise.

        Keyword arguments:
        algo -- the filename of the algorithm to be compiled
        """

        self.algo = algo
        os.system("make %s" % algo)

    def postprocess(self, data, col='y_pred', output='y_post', params=[12, 10, 1, 10, 5]):
        """Postprocess the score_data using the algorithm and output the result as score_data

        Keyword arguments:
        data -- the score_data to be postprocessed
        col -- the name of the Dataframe column to be postprocessed
        output -- the name of the Dataframe column for outputting the result
        params -- list of parameters to be input into the algorithm
        """

        ypred = data.to_binary(col)

        f = open('tmp_post.txt',"w")
        f.write(str(len(ypred))+"\n")

        for i in ypred:
            for j in range(128):
                f.write('%d' % i[j])
                if j==127:
                    f.write("\n")
                else:
                    f.write(' ')
        f.close()
        cmd = './%s tmp_post.txt' % self.algo
        for i in params:
            cmd += ' %d' % i
        os.system(cmd)
        a = []
        with open("after.txt") as f:
            for line in f:
                s = line.split()
                tmp = [int(x) for x in s]
                a.append(tmp)
        v = np.array(a)
        post = data.merge_binary(v, output)
        return post
