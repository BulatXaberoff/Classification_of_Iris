using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    public class Matrix
    {
        public double[,] data;
        public int m;
        public int n;

        public int M => this.m;

        public int N => this.n;

        public Matrix(int m, int n)
        {
            this.m = m;
            this.n = n;
            var newarr= new double[m, n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    newarr[i, j] = 0;
                }
            }
            this.data = newarr;

        }
        public Matrix(double[] arr)
        {
            m = 1;
            n = arr.Length;
            double[,] newarr = new double[m, n];
            for (int i = 0; i < n; i++)
            {
                newarr[0, i] = arr[i];
            }
            data = newarr;
        }
        public Matrix(double[,] arr)
        {
            this.m = arr.GetLength(0);
            this.n = arr.GetLength(1);
            double[,] newarr = new double[m, n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    newarr[i, j] = arr[i, j];
                }
            }
            this.data = newarr;
        }

        public double this[int x, int y]
        {
            get => this.data[x, y];
            set => this.data[x, y] = value;
        }

        public static Matrix operator *(Matrix matrix, Matrix matrix2)
        {
            if (matrix.N != matrix2.M)
                throw new ArgumentException("matrixes can not be multiplied");
            Matrix result = new Matrix(matrix.M, matrix2.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) =>
            {
                for (int index = 0; index < matrix.N; ++index)
                    result[i, j] += matrix[i, index] * matrix2[index, j];
            }));
            return result;
        }
        public static Matrix operator -(int Const, Matrix matrix1)
        {
            Matrix result = new Matrix(matrix1.M, matrix1.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) =>
            {
                for (int index = 0; index < matrix1.N; ++index)
                    result[i, j] = Const - matrix1[i, j];
            }));
            return result;
        }
        public static Matrix operator -(Matrix matrix1, Matrix matrix2)
        {
            Matrix result = new Matrix(matrix1.M, matrix1.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) =>
            {
                for (int index = 0; index < matrix1.N; index++)
                    result[i, j] = matrix1[i, j]-matrix2[i,j];
            }));
            return result;
        }

        public static Matrix operator -( Matrix matrix1, int Const)
        {
            Matrix result = new Matrix(matrix1.M, matrix1.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) =>
            {
                for (int index = 0; index < matrix1.N; ++index)
                    result[i, j] = matrix1[i, j] - Const;
            }));
            return result;
        }
        public static Matrix operator *(Matrix matrix, double value)
        {
            Matrix result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) => result[i, j] = matrix[i, j] * value));
            return result;
        }
        public static Matrix operator *( double value, Matrix matrix)
        {
            Matrix result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) => result[i, j] = value*matrix[i, j] ));
            return result;
        }
        public static Matrix operator &(Matrix matrix1, Matrix matrix2)
        {
            Matrix result = new Matrix(matrix1.M, matrix1.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) => result[i, j] = matrix1[i, j] * matrix2[i,j]));
            return result;
        }


        public static Matrix operator /(Matrix matrix, double value)
        {
            Matrix result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((Action<int, int>)((i, j) => result[i, j] = matrix[i, j] / value));
            return result;
        }

        public void ProcessFunctionOverData(Action<int, int> func)
        {
            for (int index1 = 0; index1 < this.M; ++index1)
            {
                for (int index2 = 0; index2 < this.N; ++index2)
                    func(index1, index2);
            }
        }

        public Matrix Copy()
        {
            Matrix matrix = new Matrix(this.n, this.m);
            for (int index1 = 0; index1 < this.n; ++index1)
            {
                for (int index2 = 0; index2 < this.m; ++index2)
                    matrix.data[index1, index2] = this.data[index1, index2];
            }
            return matrix;
        }

        public bool IsSquare => this.M == this.N;

        public double Determinant()
        {
            if (!this.IsSquare)
                throw new InvalidOperationException("determinant can be calculated only for square matrix");
            if (this.N == 2)
                return this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0];
            double num = 0.0;
            for (int index = 0; index < this.N; ++index)
                num += (index % 2 == 1 ? 1.0 : -1.0) * this[1, index] * this.CreateMatrixWithoutColumn(index).CreateMatrixWithoutRow(1).Determinant();
            return num;
        }

        public Matrix CreateMatrixWithoutColumn(int column)
        {
            Matrix matrix = column >= 0 && column < this.N ? new Matrix(this.M, this.N - 1) : throw new ArgumentException("invalid column index");
            matrix.ProcessFunctionOverData((Action<int, int>)((i, j) => matrix[i, j] = j < column ? this[i, j] : this[i, j + 1]));
            return matrix;
        }

        public Matrix CreateMatrixWithoutRow(int row)
        {
            Matrix result = row >= 0 && row < this.M ? new Matrix(this.M - 1, this.N) : throw new ArgumentException("invalid row index");
            result.ProcessFunctionOverData((Action<int, int>)((i, j) => result[i, j] = i < row ? this[i, j] : this[i + 1, j]));
            return result;
        }

        public Matrix Transposition()
        {
            Matrix matrix = new Matrix(this.N, this.M);
            for (int index1 = 0; index1 < this.M; ++index1)
            {
                for (int index2 = 0; index2 < this.N; ++index2)
                    matrix[index2, index1] = this[index1, index2];
            }
            return matrix;
        }

        public Matrix InverseMatrix()
        {
            double num = this.Determinant();
            if (num == 0.0)
                throw new InvalidOperationException("determinant must be not 0");
            Matrix matrix = new Matrix(this.M, this.N);
            return this.MatrixOfMinorAdd(1).MatrixOfMinorAdd(2).Transposition() / num;
        }

        public Matrix MatrixOfMinorAdd(int check)
        {
            Matrix matrix1 = new Matrix(this.M, this.M);
            double num1 = 0.0;
            if (!this.IsSquare)
                throw new InvalidOperationException("determinant can be calculated only for square matrix");
            Matrix matrix2 = new Matrix(this.N, this.N);
            switch (check)
            {
                case 1:
                    num1 = 0.0;
                    if (this.N == 2)
                    {
                        for (int index1 = 0; index1 < this.n; ++index1)
                        {
                            for (int index2 = 0; index2 < this.n; ++index2)
                            {
                                Matrix matrixWithoutRow = this.CreateMatrixWithoutColumn(index2).CreateMatrixWithoutRow(index1);
                                matrix1[index1, index2] = matrixWithoutRow[0, 0];
                            }
                        }
                        return matrix1;
                    }
                    for (int index3 = 0; index3 < this.n; ++index3)
                    {
                        for (int index4 = 0; index4 < this.n; ++index4)
                        {
                            matrix2 = this.CreateMatrixWithoutColumn(index4).CreateMatrixWithoutRow(index3);
                            double num2 = this.CreateMatrixWithoutColumn(index4).CreateMatrixWithoutRow(index3).Determinant();
                            matrix1[index4, index3] = num2;
                        }
                    }
                    return matrix1;
                case 2:
                    num1 = 0.0;
                    if (this.N == 2)
                    {
                        for (int x = 0; x < this.n; ++x)
                        {
                            for (int y = 0; y < this.n; ++y)
                                matrix1[x, y] = Math.Pow(-1.0, (double)(x + y)) * this[x, y];
                        }
                        return matrix1;
                    }
                    for (int index5 = 0; index5 < this.n; ++index5)
                    {
                        for (int index6 = 0; index6 < this.n; ++index6)
                        {
                            double num3 = Math.Pow(-1.0, (double)(index5 + index6)) * this[index5, index6];
                            matrix1[index6, index5] = num3;
                        }
                    }
                    return matrix1;
                default:
                    return matrix1;
            }
        }

        public override string ToString()
        {
            string str = "";
            for (int index1 = 0; index1 < this.M; ++index1)
            {
                for (int index2 = 0; index2 < this.N; ++index2)
                    str = str + this.data[index1, index2].ToString() + " ";
                str += "\n";
            }
            return str;
        }

        static public Matrix FindSolveOverridenSLU(Matrix A, Matrix B)
        {
            Matrix aT = A.Transposition();
            Matrix aTA = aT * A;
            Matrix aTB = aT * B;
            return aTA.InverseMatrix() * aTB;
        }
        static public Matrix FindApproximationFunc(Matrix X, Matrix Y, Matrix Z)
        {
            Matrix W = new Matrix(1, X.n);
            for (int i = 0; i < X.n; i++)
            {
                W[0, i] = 1;
            }
            W = W.ToAddColumn(X.ToLog());
            W = W.ToAddColumn(Y.ToLog());
            var Q = Z.ToLog();
            return FindSolveOverridenSLU(W, Q);
        }
        public Matrix ToLog()
        {
            Matrix logMatrix = new Matrix(this.m, this.n);
            for (int i = 0; i < this.m; i++)
            {
                for (int j = 0; j < this.n; j++)
                {
                    if (this[i, j] <= 0)
                    {
                        continue;
                    }
                    logMatrix[i, j] = Math.Log(this[i, j]);
                }
            }
            return logMatrix;
        }
        public double Sum()
        {
            double sum = 0;
            this.ProcessFunctionOverData((Action<int, int>)((i, j) =>
            {
                    sum += this[i, j];
            }));
            return sum;
            
        }
        public Matrix ToAddColumn(Matrix A)
        {
            Matrix newMatrix = new Matrix(A.m + m, n);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    newMatrix[i, j] = this[i, j];
                }
            }
            for (int i = m; i < A.m + m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    newMatrix[i, j] = A[i - m, j];
                }
            }
            //for (int i = 0; i < n; i++)
            //{
            //    newMatrix[0, i] = this[0, i];
            //}
            //for (int i = 0; i < n; i++)
            //{
            //    newMatrix[1, i] = A[0, i];
            //}
            return newMatrix;
        }
        public void Fill(int num)
        {
            Random rand = new Random();
            for (int i = 0; i < this.m; i++)
            {
                for (int j = 0; j < this.n; j++)
                {
                    this[i, j] = Convert.ToDouble(rand.Next(0, 100)) / num;
                }
            }
        }
        public double Max()
        {
            double max = 0;
            ProcessFunctionOverData((Action<int, int>)((i, j) =>
            {
                for (int index = 0; index < this.N; ++index)
                    if (this[i,j]>max)
                        max = this[i, j];
            }));
            return max;
        }

    }
}
