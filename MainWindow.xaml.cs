using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;
using System.Diagnostics;
using System.Reflection;
using System.Windows.Threading;
using System.Globalization;

namespace MachineLearning {
    /// <summary>
    /// MainWindow.xaml の相互作用ロジック
    /// </summary>
    public partial class MainWindow : Window {
        Network theNetwork;

        int TrainCnt;
        int ImgH;
        int ImgW;
        int ImgIdx = 0;


        DispatcherTimer PaintTimer;

        public MainWindow() {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e) {
            TestIsLetter();
            //Sys.TestRandomSampling();

            string path = Assembly.GetExecutingAssembly().Location;
            for(int i = 0; i < 3; i++) {
                path = System.IO.Path.GetDirectoryName(path);
            }

            string mnist_path = path + "\\MNIST";

            if (Sys.isCNN) {

                theNetwork = new Network(new Layer[] {
                    new InputLayer(28, 28),
                    //new ConvolutionalLayer(5, 20),
                    new ConvolutionalLayer(10, 2),
                    new PoolingLayer(2),
                    new FullyConnectedLayer(30),
                    new FullyConnectedLayer(10)
                });
            }
            else {

                theNetwork = new Network(new Layer[] {
                    new InputLayer(28, 28),
                    new FullyConnectedLayer(30),
                    new FullyConnectedLayer(10)
                });
            }

            byte[] buf;

            //-------------------------------------------------- TrainImage
            buf = File.ReadAllBytes(mnist_path + "\\train-images.idx3-ubyte");
            Debug.WriteLine("{0} {1} {2} {3}", BytesToInt(buf, 0), TrainCnt, ImgW, ImgH);

            TrainCnt = BytesToInt(buf, 4);
            ImgH = BytesToInt(buf, 8);
            ImgW = BytesToInt(buf, 12);

            theNetwork.TrainImage = new byte[TrainCnt, ImgH * ImgW];
            Buffer.BlockCopy(buf, 16, theNetwork.TrainImage, 0, TrainCnt * ImgH * ImgW);

            //-------------------------------------------------- TrainLabel
            buf = File.ReadAllBytes(mnist_path + "\\train-labels.idx1-ubyte");
            Debug.WriteLine("{0} {1}"        , BytesToInt(buf, 0), BytesToInt(buf, 4));

            theNetwork.TrainLabel = new byte[TrainCnt];
            Buffer.BlockCopy(buf, 8, theNetwork.TrainLabel, 0, TrainCnt);

            //-------------------------------------------------- TestImage
            buf = File.ReadAllBytes(mnist_path + "\\t10k-images.idx3-ubyte");
            Debug.WriteLine("{0} {1} {2} {3}", BytesToInt(buf, 0), BytesToInt(buf, 4), BytesToInt(buf, 8), BytesToInt(buf, 12));

            int test_cnt = BytesToInt(buf, 4);
            theNetwork.TestImage = new byte[test_cnt, ImgH * ImgW];
            Buffer.BlockCopy(buf, 16, theNetwork.TestImage, 0, test_cnt * ImgH * ImgW);

            //-------------------------------------------------- TestLabel
            buf = File.ReadAllBytes(mnist_path + "\\t10k-labels.idx1-ubyte");
            Debug.WriteLine("{0} {1}"        , BytesToInt(buf, 0), BytesToInt(buf, 4));

            theNetwork.TestLabel = new byte[test_cnt];
            Buffer.BlockCopy(buf, 8, theNetwork.TestLabel, 0, test_cnt);

            Task.Run(() => {
                if (Sys.isDebug) {

                    theNetwork.SGD(30, 1, 3.0);
                }
                else {

                    //theNetwork.SGD(30, 11, 3.0);
                    //theNetwork.SGD(30, 12, 10.0);
                    theNetwork.SGD(30, 10, 10.0);
                }
            });

            PaintTimer = new DispatcherTimer(DispatcherPriority.Normal);
            PaintTimer.Interval = TimeSpan.FromMilliseconds(10 * 1000);
            PaintTimer.Tick += PaintTimer_Tick;
            PaintTimer.Start();
        }

        private void PaintTimer_Tick(object sender, EventArgs e) {
            int wh = ImgW * ImgH;

            byte[] vb = new byte[wh];
            Buffer.BlockCopy(theNetwork.TrainImage, wh * ImgIdx, vb, 0, wh);
            BitmapSource bmp = BitmapSource.Create(ImgW, ImgH, 96, 96, PixelFormats.Gray8, null, vb, ImgW);

            //ImageSource old_img = img_Input.Source;
            img_Input.Source = bmp;

            ImgIdx++;
            if(TrainCnt <= ImgIdx) {

                PaintTimer.Stop();
            }
        }

        int BytesToInt(byte[] v, int offset) {
            return v[offset] * 0x1000000 + v[offset + 1] * 0x10000 + v[offset + 2] * 0x100 + v[offset + 3];
        }

        public void TestIsLetter() {
            bool is_letter = false, is_letter_sv = false;
            int start = -1;
            int k = 0;
            Dictionary<UnicodeCategory, StringWriter> dic = new Dictionary<UnicodeCategory, StringWriter>();

            StringWriter sw1 = new StringWriter();
            StringWriter sw2 = new StringWriter();
            sw1.Write("\t[\r\n\t");
            sw2.Write("\t[\r\n\t");
            for (int i = 0; i < 0xFFFF; i++) {
                char c = (char)i;
                UnicodeCategory cat = Char.GetUnicodeCategory(c);
                is_letter = Char.IsLetter(c);
/*

                if (is_letter) {
                    if(! dic.TryGetValue(cat, out sw2)) {
                        sw2 = new StringWriter();
                        dic.Add(cat, sw2);
                    }
                    sw2.Write(c);

                }

                if(start == -1) {

                    if (is_letter) {

                        start = i;
                        sw1 = new StringWriter();
                        sw1.Write(c);
                        k = 1;
                    }
                }
                else {

                    if (is_letter) {

                        k++;
                        sw1.Write(c);
                        if(k == 200) {
                            sw1.Write("\r\n");
                            k = 0;
                        }
                    }
                    else {

                        if (cat != UnicodeCategory.OtherNotAssigned) {

                            Debug.WriteLine("{0:X} - {1:X}", start, i - 1);
                            //Debug.WriteLine(sw1.ToString());

                            start = -1;
                        }
                    }
                }
*/

                if (is_letter != is_letter_sv) {
                    if (is_letter) {
                        start = i;
                    }
                    else {
                        sw1.Write(string.Format("0x{0:X}", start));
                        sw2.Write(string.Format("0x{0:X}", i));
                        k++;
                        if (k == 20) {

                            k = 0;
                            sw1.Write(",\r\n\t");
                            sw2.Write(",\r\n\t");
                        }
                        else {
                            sw1.Write(", ");
                            sw2.Write(", ");
                        }
                    }

                    is_letter_sv = is_letter;
                }
            }

            sw1.WriteLine("\r\n\t]");
            sw2.WriteLine("\r\n\t]");

/*
            foreach (UnicodeCategory cat1 in dic.Keys) {

                //Debug.WriteLine("{0} {1}", cat1, dic[cat].ToString());
                Debug.WriteLine("{0}", cat1);
            }
*/


            Debug.WriteLine(sw1.ToString());
            Debug.WriteLine("--------------------------------------------------------------");
            Debug.WriteLine(sw2.ToString());
        }
    }

    public struct TFPair {
        public float X;
        public float Y;

        
    }

    public class TUtil {
        public static float Sigmoid(float z) {
            return (float)(1.0f / (1 + Math.Pow(Math.E, -z)));
        }

        public static float SigmoidPrime(float z) {
            float f = Sigmoid(z);

            return f * (1 - f);
        }

        public static float[] CostDerivative(float[] v, float[] y) {
            return (from p in Zip(v,y) select p.X - p.Y).ToArray();
        }

        public static IEnumerable<TFPair> Zip(float[] v, float[] w) {
            for(int i = 0; i < v.Length; i++) {
                TFPair p;

                p.X = v[i];
                p.Y = w[i];

                yield return p;
            }
        }
    }

    public class TNormalRandom {
        bool Flag = false;
        double C;
        double Theta;

        public static Random Rn = new Random(System.Environment.TickCount);
        public static TNormalRandom NormalRandom = new TNormalRandom();

        public double NextDouble() {
            Flag = !Flag;
            if (Flag) {
                C = Math.Sqrt(-2 * Math.Log(Rn.NextDouble()));
                Theta = Rn.NextDouble() * Math.PI * 2;

                return C * Math.Sin(Theta);
            }
            else {
                return C * Math.Cos(Theta);
            }
        }

        public static void Test() {
            int[] v = new int[200];
            for (int i = 0; i < 1000000; i++) {
                int k = (int)(NormalRandom.NextDouble() * 25 + v.Length / 2);
                if (0 <= k && k < v.Length) {
                    v[k]++;
                }
            }
            for (int k = 0; k < v.Length; k++) {
                Debug.WriteLine("{0}", v[k]);
            }

            Array2 m1 = new Array2(2, 3), m2 = new Array2(2, 3), m3 = new Array2(2, 3);

            m3 = m1 + m2;

            m3 = m1.Apply((double f1, double f2) => f1 * f2, m2);
        }

        public static Array2 randn(int rows, int cols) {
            Array2 m = new Array2(rows, cols);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    m[r, c] = NormalRandom.NextDouble();
                }
            }

            return m;
        }

        public static Array1 randn(int rows) {
            Array1 m = new Array1(rows);

            for (int r = 0; r < rows; r++) {
                m[r] = NormalRandom.NextDouble();
            }

            return m;
        }

    }

    public delegate double F0();
    public delegate double F1(double f1);
    public delegate double F2(double f1, double f2);

    public class ArrayN {
        public virtual Array GetData() {
            Debug.Assert(false, "Array-N-Get-Data");
            return null;
        }

        public int[] Shape() {
            Array dt = GetData();

            switch (dt.Rank) {
            case 4:
                return new int[] { dt.GetLength(0), dt.GetLength(1), dt.GetLength(2), dt.GetLength(3) };
            case 3:
                return new int[] { dt.GetLength(0), dt.GetLength(1), dt.GetLength(2) };
            case 2:
                return new int[] { dt.GetLength(0), dt.GetLength(1) };
            case 1:
                return new int[] { dt.GetLength(0) };

            default:
                Debug.Assert(false);
                return null;
            }
        }

        public ArrayN Reshape(params int[] args) {
            Array src = GetData();

            Array dst;
            switch (args.Length) {
            case 1:
                dst = new double[args[0]];
                break;

            case 2:
                dst = new double[args[0], args[1]];
                break;

            case 3:
                dst = new double[args[0], args[1], args[2]];
                break;

            case 4:
                dst = new double[args[0], args[1], args[2], args[3]];
                break;

            default:
                Debug.Assert(false, "Array-N-Get-Data");
                return null;
            }

            Debug.Assert(src.Length == dst.Length, "Array-N-Get-Data");
            Buffer.BlockCopy(src, 0, dst, 0, sizeof(double) * src.Length);

            switch (args.Length) {
            case 1:
                return new Array1((double[])dst);

            case 2:
                return new Array2((double[,])dst);

            case 3:
                return new Array3((double[,,])dst);

            case 4:
                return new Array4((double[,,,])dst);
            }

            return null;
        }
    }

    public class Array1 : ArrayN {
        public double[] dt;
        public int Length;

        public Array1(int len) {
            dt = new double[len];
            Length = dt.Length;
        }

        public Array1(IEnumerable<double> v) {
            dt = v.ToArray();
            Length = dt.Length;
        }

        public Array1 Clone() {
            return new Array1((double[])dt.Clone());
        }

        public override Array GetData() {
            return dt;
        }

        public double this[int i] {
            set { dt[i] = value; }
            get { return dt[i]; }
        }

        public static Array1 operator +(Array1 a, Array1 b) {
            return new Array1(from i in Enumerable.Range(0, a.Length) select a.dt[i] + b.dt[i]);
        }

        public static Array1 operator -(Array1 a, Array1 b) {
            return new Array1(from i in Enumerable.Range(0, a.Length) select a.dt[i] - b.dt[i]);
        }

        public static Array1 operator *(Array1 a, Array1 b) {
            return new Array1(from i in Enumerable.Range(0, a.Length) select a.dt[i] * b.dt[i]);
        }

        public static Array1 operator *(double a, Array1 v) {
            return new Array1(from x in v.dt select a * x);
        }

        public Array1 Map(F1 fnc) {
            return new Array1( from x in dt select fnc(x) );
        }

        public Array1 Apply(F2 fnc, Array1 m1) {
            Array1 m3 = new Array1(Length);

            for (int r = 0; r < Length; r++) {
                m3.dt[r] = fnc(dt[r], m1.dt[r]);
            }

            return m3;
        }

        public double Reduce(F2 fnc) {
            double x = dt[0];

            for (int i = 1; i < Length; i++) {
                x = fnc(x, dt[i]);
            }

            return x;
        }

        public double Sum() {
            return Reduce((double x, double y) => x + y);
        }

        public double Max() {
            double max = Double.MinValue;

            for (int r = 0; r < Length; r++) {
                max = Math.Max(max, dt[r]);
            }

            return max;
        }

        public int ArgMax() {
            double max_val = dt[0];
            int max_idx = 0;

            for(int i = 1; i < Length; i++) {
                if(max_val < dt[i]) {

                    max_val = dt[i];
                    max_idx = i;
                }
            }

            return max_idx;
        }

        public override string ToString() {
            StringWriter sw = new StringWriter();

            for (int i = 0; i < Length; i++) {
                sw.Write("\t{0}", dt[i]);
            }
            sw.WriteLine();

            return sw.ToString();
        }
    }

    public class Array2 : ArrayN {
        public double[,] dt;

        public int nRow;
        public int nCol;

        public Array2(int rows, int cols) {
            dt = new double[rows, cols];
            nRow = dt.GetLength(0);
            nCol = dt.GetLength(1);
        }

        public Array2(double[,] init) {
            dt = init;
            nRow = dt.GetLength(0);
            nCol = dt.GetLength(1);
        }

        public Array2 Clone() {
            return new Array2((double[,])dt.Clone());
        }

        public override Array GetData() {
            return dt;
        }

        public double this[int i, int j] {
            set { dt[i, j] = value; }
            get { return dt[i, j]; }
        }

        public Array2 T() {
            Array2 m = new Array2(nCol, nRow);

            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    m.dt[c, r] = dt[r, c];
                }
            }

            return m;
        }

        public static Array2 operator +(Array2 a, Array2 b) {
            Debug.Assert(Enumerable.SequenceEqual(a.Shape(), b.Shape()), "array-2 +");

            Array2 m = new Array2(a.nRow, a.nCol);
            for (int r = 0; r < a.nRow; r++) {
                for (int c = 0; c < a.nCol; c++) {
                    m.dt[r, c] = a.dt[r, c] + b.dt[r, c];
                }
            }

            return m;
        }

        public static Array2 operator -(Array2 a, Array2 b) {
            Debug.Assert(Enumerable.SequenceEqual(a.Shape(), b.Shape()), "array-2 -");

            Array2 m = new Array2(a.nRow, a.nCol);
            for (int r = 0; r < a.nRow; r++) {
                for (int c = 0; c < a.nCol; c++) {
                    m.dt[r, c] = a.dt[r, c] - b.dt[r, c];
                }
            }

            return m;
        }

        public static Array2 operator *(Array2 a, Array2 b) {
            Debug.Assert(Enumerable.SequenceEqual(a.Shape(), b.Shape()), "array-2 *");

            Array2 m = new Array2(a.nRow, a.nCol);
            for (int r = 0; r < a.nRow; r++) {
                for (int c = 0; c < a.nCol; c++) {
                    m.dt[r, c] = a.dt[r, c] * b.dt[r, c];
                }
            }

            return m;
        }


        public static Array2 operator +(Array2 a, Array1 b) {
            Debug.Assert(a.nCol == b.Length, "array-2 +");

            Array2 m = new Array2(a.nRow, a.nCol);
            for (int r = 0; r < a.nRow; r++) {
                for (int c = 0; c < a.nCol; c++) {
                    m.dt[r, c] = a.dt[r, c] + b.dt[c];
                }
            }

            return m;
        }


        public static Array2 operator *(double a, Array2 b) {
            Array2 m = new Array2(b.nRow, b.nCol);

            for (int r = 0; r < b.nRow; r++) {
                for (int c = 0; c < b.nCol; c++) {
                    m.dt[r, c] = a * b.dt[r, c];
                }
            }

            return m;
        }

        public Array2 Dot(Array2 m) {
            Debug.Assert(nCol == m.nRow, "ndarray-2-dot");
            Array2 ret = new Array2(nRow, m.nCol);

            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < m.nCol; c++) {
                    double sum = 0;
                    for (int k = 0; k < nCol; k++) {
                        sum += dt[r, k] * m.dt[k, c];
                    }
                    ret.dt[r, c] = sum;
                }
            }

            return ret;
        }

        public Array1 Row(int r) {
            return new Array1(from c in Enumerable.Range(0, nCol) select dt[r, c]);
        }

        public Array1 Col(int c) {
            return new Array1(from r in Enumerable.Range(0, nRow) select dt[r, c]);
        }

        public Array1[] Rows() {
            return (from r in Enumerable.Range(0, nRow) select new Array1(from c in Enumerable.Range(0, nCol) select dt[r, c])).ToArray();
        }

        public Array1[] Cols() {
            return (from c in Enumerable.Range(0, nCol) select new Array1(from r in Enumerable.Range(0, nRow) select dt[r, c])).ToArray();
        }

        public Array2 Map(F1 fnc) {
            Array2 ret = new Array2(nRow, nCol);

            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    ret.dt[r, c] = fnc(dt[r, c]);
                }
            }

            return ret;
        }

        public Array2 Apply(F2 fnc, Array2 m1) {
            Array2 m3 = new Array2(nRow, nCol);

            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    m3.dt[r, c] = fnc(dt[r, c], m1.dt[r, c]);
                }
            }

            return m3;
        }

        public double Sum() {
            double sum = 0;
            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    sum += dt[r, c];
                }
            }

            return sum;
        }

        public double Max() {
            double max = Double.MinValue;

            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    max = Math.Max(max, dt[r, c]);
                }
            }

            return max;
        }

        public Array1 SumRow() {
            Array1 m = new Array1(nRow);

            for (int r = 0; r < nRow; r++) {
                double sum = 0;
                for (int c = 0; c < nCol; c++) {
                    sum += dt[r, c];
                }
                m.dt[r] = sum;
            }

            return m;
        }

        public override string ToString() {
            StringWriter sw = new StringWriter();

            for (int r = 0; r < nRow; r++) {
                for (int c = 0; c < nCol; c++) {
                    sw.Write("\t{0}", dt[r, c]);
                }
                sw.WriteLine();
            }

            return sw.ToString();
        }
    }

    public class Array3 : ArrayN {
        public double[,,] dt;
        public int nDepth;
        public int nRow;
        public int nCol;

        public Array3(int depth, int rows, int cols) {
            dt = new double[depth, rows, cols];
            nDepth = dt.GetLength(0);
            nRow = dt.GetLength(1);
            nCol = dt.GetLength(2);
        }

        public Array3(double[,,] init) {
            dt = init;
            nDepth = dt.GetLength(0);
            nRow = dt.GetLength(1);
            nCol = dt.GetLength(2);
        }

        public override Array GetData() {
            return dt;
        }

        public double this[int i, int j, int k] {
            set { dt[i, j, k] = value; }
            get { return dt[i, j, k]; }
        }

        public static Array3 operator +(Array3 a, Array3 b) {
            Debug.Assert(Enumerable.SequenceEqual(a.Shape(), b.Shape()), "array-3 +");

            Array3 m = new Array3(a.nDepth, a.nRow, a.nCol);
            for(int d = 0; d < a.nDepth; d++) {
                for (int r = 0; r < a.nRow; r++) {
                    for (int c = 0; c < a.nCol; c++) {
                        m.dt[d, r, c] = a.dt[d, r, c] + b.dt[d, r, c];
                    }
                }
            }

            return m;
        }


        public static Array3 operator *(double a, Array3 b) {
            Array3 m = new Array3(b.nDepth, b.nRow, b.nCol);

            for (int d = 0; d < b.nDepth; d++) {
                for (int r = 0; r < b.nRow; r++) {
                    for (int c = 0; c < b.nCol; c++) {
                        m.dt[d, r, c] = a * b.dt[d, r, c];
                    }
                }
            }

            return m;
        }

        public Array1 Depth(int r, int c) {
            Array1 m = new Array1(nDepth);

            for (int d = 0; d < nDepth; d++) {
                m.dt[d] = dt[d, r, c];
            }

            return m;
        }

        public Array3 Set(F0 fnc) {
            for (int d = 0; d < nDepth; d++) {
                for (int r = 0; r < nRow; r++) {
                    for (int c = 0; c < nCol; c++) {
                        dt[d, r, c] = fnc();
                    }
                }
            }

            return this;
        }


        public Array3 Map(F1 fnc) {
            Array3 m3 = new Array3(nDepth, nRow, nCol);

            for (int d = 0; d < nDepth; d++) {
                for (int r = 0; r < nRow; r++) {
                    for (int c = 0; c < nCol; c++) {
                        m3.dt[d, r, c] = fnc(dt[d, r, c]);
                    }
                }
            }

            return m3;
        }

        public Array3 Apply(F2 fnc, Array3 m1) {
            Array3 m3 = new Array3(nDepth, nRow, nCol);

            for (int d = 0; d < nDepth; d++) {
                for (int r = 0; r < nRow; r++) {
                    for (int c = 0; c < nCol; c++) {
                        m3.dt[d, r, c] = fnc(dt[d, r, c], m1.dt[d, r, c]);
                    }
                }
            }

            return m3;
        }

        public double Sum() {
            double sum = 0;
            for (int d = 0; d < nDepth; d++) {
                for (int r = 0; r < nRow; r++) {
                    for (int c = 0; c < nCol; c++) {
                        sum += dt[d, r, c];
                    }
                }
            }

            return sum;
        }

        public double Max() {
            double max = Double.MinValue;

            for (int d = 0; d < nDepth; d++) {
                for (int r = 0; r < nRow; r++) {
                    for (int c = 0; c < nCol; c++) {
                        max = Math.Max(max, dt[d, r, c]);
                    }
                }
            }

            return max;
        }

        public override string ToString() {
            StringWriter sw = new StringWriter();

            for (int d = 0; d < nDepth; d++) {
                for (int r = 0; r < nRow; r++) {
                    for (int c = 0; c < nCol; c++) {
                        sw.Write(",{0}", dt[d, r, c]);
                    }
                    sw.Write(", ");
                }

                sw.WriteLine();
            }

            return sw.ToString();
        }
    }

    public class Array4 : ArrayN {
        public double[,,,] dt;

        public Array4(int n1, int n2, int n3, int n4) {
            dt = new double[n1, n2, n3, n4];
        }

        public Array4(int[] shape) {
            dt = new double[shape[0], shape[1], shape[2], shape[3]];
        }

        public Array4(double[,,,] init) {
            dt = init;
        }

        public Array4 Clone() {
            return new Array4((double[,,,])dt.Clone());
        }

        public override Array GetData() {
            return dt;
        }

        public double this[int i, int j, int k, int l] {
            set { dt[i, j, k, l] = value; }
            get { return dt[i, j, k, l]; }
        }

        public Array1 At3(int j, int k, int l) {
            int n0 = dt.GetLength(0);
            Array1 m = new Array1(n0);

            for(int i = 0; i < n0; i++) {
                m.dt[i] = dt[i, j, k, l];
            }

            return m;
        }

        public void Set3(int j, int k, int l, Array1 m) {
            int n0 = dt.GetLength(0);

            for (int i = 0; i < n0; i++) {
                dt[i, j, k, l] = m.dt[i];
            }
        }

        public static Array4 operator +(Array4 a, Array4 b) {
            Debug.Assert(Enumerable.SequenceEqual(a.Shape(), b.Shape()), "Array-4 +");
            int n0 = a.dt.GetLength(0);
            int n1 = a.dt.GetLength(1);
            int n2 = a.dt.GetLength(2);
            int n3 = a.dt.GetLength(3);

            Array4 m = new Array4(n0, n1, n2, n3);

            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                            m.dt[i, j, k, l] = a.dt[i, j, k, l] + b.dt[i, j, k, l];
                        }
                    }
                }
            }

            return m;
        }

        public static Array4 operator *(Array4 a, Array4 b) {
            Debug.Assert(Enumerable.SequenceEqual(a.Shape(), b.Shape()), "Array-4 *");
            int n0 = a.dt.GetLength(0);
            int n1 = a.dt.GetLength(1);
            int n2 = a.dt.GetLength(2);
            int n3 = a.dt.GetLength(3);

            Array4 m = new Array4(n0, n1, n2, n3);

            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                            m.dt[i, j, k, l] = a.dt[i, j, k, l] * b.dt[i, j, k, l];
                        }
                    }
                }
            }

            return m;
        }

        public Array4 Map(F1 fnc) {
            int n0 = dt.GetLength(0);
            int n1 = dt.GetLength(1);
            int n2 = dt.GetLength(2);
            int n3 = dt.GetLength(3);

            Array4 m = new Array4(n0, n1, n2, n3);

            for(int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                            m.dt[i, j, k, l] = fnc(dt[i, j, k, l]);
                        }
                    }
                }
            }

            return m;
        }

        public Array4 Apply(F2 fnc, Array4 m1) {
            int n0 = dt.GetLength(0);
            int n1 = dt.GetLength(1);
            int n2 = dt.GetLength(2);
            int n3 = dt.GetLength(3);

            Array4 m3 = new Array4(Shape());

            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                            m3.dt[i, j, k, l] = fnc(dt[i, j, k, l], m1.dt[i, j, k, l]);
                        }
                    }
                }
            }

            return m3;
        }

        public double Max() {
            int n0 = dt.GetLength(0);
            int n1 = dt.GetLength(1);
            int n2 = dt.GetLength(2);
            int n3 = dt.GetLength(3);
            double max = Double.MinValue;

            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                            max = Math.Max(max, dt[i, j, k, l]);
                        }
                    }
                }
            }

            return max;
        }

        public static Array4 operator -(Array4 a, Array4 b) {
            return a.Apply((x, y) => x - y, b);
        }

        public override string ToString() {
            StringWriter sw = new StringWriter();
            int n0 = dt.GetLength(0);
            int n1 = dt.GetLength(1);
            int n2 = dt.GetLength(2);
            int n3 = dt.GetLength(3);

            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    for (int k = 0; k < n2; k++) {
                        for (int l = 0; l < n3; l++) {
                            sw.Write(",{0}", dt[i, j, k, l]);
                        }
                        sw.Write(", ");
                    }
                    sw.WriteLine();
                }
                sw.WriteLine();
            }

            return sw.ToString();
        }
    }
}

