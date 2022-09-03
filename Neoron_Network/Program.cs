using Neural_Network;
Random rnd=new Random();

const int INPUT_Layer = 4;
const int HIDDEN_Layer_1 = 8;
//const int HIDDEN_Layer_2 = 6;
const int OUTPUT_Layer = 3;
double Alpha = 0.001;

var X_Data = new Matrix(1, INPUT_Layer);
X_Data.Fill(10);
var Y_Data = new Matrix(1, OUTPUT_Layer);
Y_Data.Fill(10);
var W1 = new Matrix(INPUT_Layer, HIDDEN_Layer_1);
W1.Fill(100);
var W2 = new Matrix(HIDDEN_Layer_1, OUTPUT_Layer);
W2.Fill(100);

Read_Csv(ref X_Data, ref Y_Data);

void Drop_Data(out Matrix X,out Matrix Y,int index)
{
    double[,] arrx = new double[1, INPUT_Layer];
    double[,] arry = new double[1, OUTPUT_Layer];
    for (int i = 0; i < INPUT_Layer; i++)
    {
        arrx[0, i] = X_Data[index,i];
    }
    for (int i = 0; i < OUTPUT_Layer; i++)
    {
        arry[0,i]=Y_Data[index,i];
    }
    X = new Matrix(arrx);
    Y = new Matrix(arry);
}

for (int i = 0; i < 10000; i++)
{
    int index = rnd.Next(0, 149);
    Matrix X, Y;
    Drop_Data(out X,out Y, index);


    var T1 = (X * W1);
    var H1 = Relu(T1);
    var T2 = (H1 * W2);
    var Z = Soft_Max(T2);
    var E = Cross_Entropy(Z, Y);

    //Backward
    var dEdT2 = Cross_Cross_Entropy_Deriv(Z, Y);
    var dEdW2 = H1.Transposition() * dEdT2;

    var dEdH1 = dEdT2 * W2.Transposition();
    var dEdT1 = dEdH1 & Relu_Deriv(T1);
    var dEdW1 = X.Transposition() * dEdT1;

    W2 -= Alpha * dEdW2;
    W1 -= Alpha * dEdW1;

}

#region

Matrix Relu(Matrix x)
{
    Matrix t = new Matrix(x.m,x.n);
    for (int i = 0; i < x.M; i++)
    {
        for (int j = 0; j < x.N; j++)
        {
            t[i, j] = Math.Max(0, x.data[i, j]);
        }
    }
    return t;
}
Matrix Relu_Deriv(Matrix x)
{
    Matrix t = new Matrix(x.m, x.n);
    for (int i = 0; i < x.M; i++)
    {
        for (int j = 0; j < x.N; j++)
        {
            if (x[i, j] >= 0)
            {
                t[i, j] = 1;
            }
            t[i,j] = 0;
        }
    }
    return t;
}

Matrix Sigma(Matrix x)
{
    Matrix t = new Matrix(x.m, x.n);
    for (int i = 0; i < x.M; i++)
    {
        for (int j = 0; j < x.N; j++)
        {
            var r = -x[i, j];
            t[i,j]=1 / (1 + Math.Exp(r));
        }
    }
    return t;
}
Matrix Sigma_Deriv(Matrix x)
{
    return Sigma(x) & (1 - Sigma(x));
}
Matrix Soft_Max(Matrix pred)
{
    Matrix t = new Matrix(pred.m, pred.n);
    var size = OUTPUT_Layer;
    double sum = 0;
    for (int j = 0; j < size; j++)
    {
        sum += Math.Exp(pred[0, j]);
    }
    for (int j = 0; j < size; j++)
    {
        t[0,j] = Math.Exp(pred[0,j]) / sum;
    }
    return t;
}

Matrix Cross_Entropy(Matrix pred, Matrix y)
{
    var size = OUTPUT_Layer;
    Matrix t = new Matrix(pred.m, pred.n);
    for (int j = 0; j < size; j++)
    {
        t[0,j] = -Math.Log(pred[0,j]);
    }
    return t;
}

Matrix Cross_Cross_Entropy_Deriv(Matrix pred, Matrix y)
{
    return pred-y;
}
double ArgMax(Matrix x)
{
    return x.Max();
}
//void Read_CSV(string filename,out X,out Y)
//{
//    StreamReader sr = new StreamReader(filename);
//    string line;
//    string[] row = new string[5];
//    while ((line = sr.ReadLine()) != null)
//    {
//        row = line.Split(',');
//        X=r
//    }
//}
#endregion
static void Read_Csv(ref Matrix X,ref Matrix Y)
{
    var filename = @"C:\Users\bulyn\source\repos\Neoron_Network\Neoron_Network\Data\Iris.csv";
    StreamReader sr = new StreamReader(filename);
    string line;
    string[] row = new string[6];
    int c = 0;
    double[,] arrx = new double[150, 4];
    double[,] arry = new double[150, 3];
    while ((line = sr.ReadLine()) != null)
    {
        row = line.Split('.');
        if (c == 0)
        {
            c++;
            continue;
        }
        for (int i = 0; i < arrx.GetLength(1); i++)
        {
            arrx[c - 1, i] = Convert.ToDouble(row[i + 1]);
        }
        if ("Iris-setosa" == row[5])
        {
            arry[c - 1, 0] = 1.0;
        }
        if ("Iris-versicolor" == row[5])
        {
            arry[c - 1, 1] = 1.0;
        }
        if ("Iris-virginica" == row[5])
        {
            arry[c - 1, 2] = 1.0;
        }
        c++;
    }
    X = new Matrix(arrx);
    Y = new Matrix(arry);
}
double Predict(Matrix X)
{
    var T1 = (X * W1);
    var H1 = Relu(T1);
    var T2 = (H1 * W2);
    var Z = Soft_Max(T2);
    return ArgMax(Z);
}
Matrix Predict1(Matrix X)
{
    var T1 = (X * W1);
    var H1 = Relu(T1);
    var T2 = (H1 * W2);
    var Z = Soft_Max(T2);
    return Z;
}
for (int i = 0; i < 20; i++)
{
    int index = rnd.Next(0, 150);
    Matrix X1, Y1;
    Drop_Data(out X1, out Y1,index);
    Console.WriteLine($"{index}->{Predict1(X1)}");

}
