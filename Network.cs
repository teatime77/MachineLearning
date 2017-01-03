using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading.Tasks;

namespace MachineLearning {
    public class Layer {
        public int FwCnt = 0;
        public double FwTime = 0;
        public int BwCnt = 0;
        public double BwTime = 0;
        public Layer PrevLayer;
        public Layer NextLayer;
        public int UnitSize;
        public int ImgRows;
        public int ImgCols;

        public Layer() {
        }

        public virtual void init(Layer prev_layer) {
            PrevLayer = prev_layer;
            if (prev_layer != null) {
                prev_layer.NextLayer = this;
            }
        }

        public virtual Array1 GetActivation1() {
            Debug.Assert(false, "layer-Get-Activation-1");
            return null;
        }

        public virtual Array2 GetActivation2() {
            Debug.Assert(false, "layer-Get-Activation-2");
            return null;
        }

        public virtual Array3 GetActivation3() {
            Debug.Assert(false, "layer-Get-Activation-3");
            return null;
        }

        public virtual Array4 GetActivation4() {
            Debug.Assert(false, "layer-Get-Activation-4");
            return null;
        }

        public virtual void SaveParam() {
        }

        public virtual void RestoreParam() {
        }

        public virtual int ParamLength() {
            return 0;
        }

        public virtual double ParamAt(int i) {
            Debug.Assert(false, "layer-Param-At");
            return 0;
        }

        public virtual void ParamSet(int i, double p) {
            Debug.Assert(false, "layer-Param-Set");
        }

        public virtual Array1 dC_dPs(int batch_idx) {
            return new Array1(0);
        }

        public virtual Array1 dC_dPByParamIdx(int param_idx) {
            return new Array1(0);
        }

        public virtual double dC_dPByParamIdxAvg(int param_idx) {
            return double.NaN;
        }

        public virtual void Forward() {
        }

        public virtual void Backward(Array2 Y) {
        }

        public void forward2() {
            DateTime startTime = DateTime.Now;
            Forward();
            FwCnt++;
            FwTime += (DateTime.Now - startTime).TotalMilliseconds;
        }

        public void backward2(Array2 Y) {
            DateTime startTime = DateTime.Now;
            Backward(Y);
            BwCnt++;
            BwTime += (DateTime.Now - startTime).TotalMilliseconds;
        }

        public virtual void UpdateParameter() {
        }
    }

    public class InputLayer : Layer {
        public Array2 Activation2;
        public Array3 Activation3;

        public InputLayer(int rows, int cols) : base() {
            ImgRows = rows;
            ImgCols = cols;
            UnitSize = rows * cols;
        }

        public override Array2 GetActivation2() {
            return Activation2;
        }

        public override Array3 GetActivation3() {
            return Activation3;
        }
    }

    public class FullyConnectedLayer : Layer {
        public Array2 Activation2;
        public Array2 Z2;
        public Array2 dC_dA2;
        public Array1 Cost;

        public Array2 svActivation2;
        public Array2 svZ2;
        public Array2 svdC_dA2;

        public Array1 Bias;
        public Array2 Weight;
        public Array1 svBias;
        public Array2 svWeight;

        public Array2 dC_dZ2;
        public Array2 NablaBiases;
        public Array3 NablaWeights;
        public Array2 svNablaBiases;
        public Array3 svNablaWeights;

        public FullyConnectedLayer(int size) : base() {

            UnitSize = size;
        }

        public override void init(Layer prev_layer) {
            base.init(prev_layer);

            Bias = TNormalRandom.randn(UnitSize);
            Weight = TNormalRandom.randn(PrevLayer.UnitSize, UnitSize);
        }

        public override Array2 GetActivation2() {
            return Activation2;
        }

        public override void SaveParam() {
            svBias = Bias.Clone();
            svWeight = Weight.Clone();
            svNablaBiases = NablaBiases.Clone();
            svNablaWeights = NablaWeights.Clone();
            svdC_dA2 = dC_dA2.Clone();
        }

        public override void RestoreParam() {
            Bias = svBias.Clone();
            Weight = svWeight.Clone();
            NablaBiases = svNablaBiases.Clone();
            NablaWeights = svNablaWeights.Clone();
            dC_dA2 = svdC_dA2.Clone();
        }

        public override int ParamLength() {
            return Bias.Length + Weight.Length;
        }

        public override double ParamAt(int i) {
            if(i < Bias.Length) {
                return Bias.dt[i];
            }
            int j = i - Bias.Length;
            int r = j / Weight.nCol;
            int c = j - r * Weight.nCol;
            return Weight.dt[r,c];
        }

        public override void ParamSet(int i, double p) {
            if (i < Bias.Length) {
                Bias.dt[i] = p;
                return;
            }
            int j = i - Bias.Length;
            int r = j / Weight.nCol;
            int c = j - r * Weight.nCol;
            Weight.dt[r, c] = p;
        }

        public override Array1 dC_dPs(int batch_idx) {
            Array1 v = new Array1(ParamLength());

            int param_i = 0;
            for (int filter_idx = 0; filter_idx < Bias.Length; filter_idx++) {
                v.dt[param_i] = NablaBiases[batch_idx, filter_idx];

                param_i++;
            }

            for (int r = 0; r < Weight.nRow; r++) {

                for (int c = 0; c < Weight.nCol; c++) {
                    v.dt[param_i] = NablaWeights[batch_idx, r, c];

                    param_i++;
                }
            }
            
            return v;
        }

        public override Array1 dC_dPByParamIdx(int param_idx) {
            int mini_batch_size = NablaBiases.nRow;

            if (param_idx < Bias.Length) {

                return new Array1(from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaBiases[batch_idx, param_idx]);
            }

            int i = param_idx - Bias.Length;
            int r = i / Weight.nCol;
            int c = i - r * Weight.nCol;

            return new Array1(from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaWeights[batch_idx, r, c]);
        }

        public override double dC_dPByParamIdxAvg(int param_idx) {
            int mini_batch_size = NablaBiases.nRow;

            if (param_idx < Bias.Length) {

                return (from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaBiases[batch_idx, param_idx]).Average();
            }

            int i = param_idx - Bias.Length;
            int r = i / Weight.nCol;
            int c = i - r * Weight.nCol;

            return (from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaWeights[batch_idx, r, c]).Average();
        }

        static Dictionary<string, double> SpanC = new Dictionary<string, double>();
        static Dictionary<string, double> SpanG = new Dictionary<string, double>();
        static Dictionary<string, int> CntC = new Dictionary<string, int>();
        static Dictionary<string, int> CntG = new Dictionary<string, int>();

        public override void Forward() {
            Array2 prev_A2 = PrevLayer.GetActivation2();

            if (Sys.CPU){

                Z2 = prev_A2.Dot(Weight) + Bias;
                Activation2 = Z2.Map(Sys.Sigmoid);
            }
            else{

                DateTime start = DateTime.Now;

                Z2 = new Array2(prev_A2.nRow, Weight.nCol);
                Activation2 = new Array2(prev_A2.nRow, Weight.nCol);
                unsafe
                {
                    fixed (double* prev_A2_dev = prev_A2.dt, Weight_dev = Weight.dt, Bias_dev = Bias.dt, z2_dev = Z2.dt, a2_dev = Activation2.dt) {
                        Array2_ prev_A2_ = new Array2_(prev_A2_dev, prev_A2);
                        Array2_ Weight_ = new Array2_(Weight_dev, Weight);
                        Array1_ Bias_ = new Array1_(Bias_dev, Bias);
                        Array2_ z2_ = new Array2_(z2_dev, Z2, false);
                        Array2_ a2_ = new Array2_(a2_dev, Activation2, false);

                        Sys.FullForward(prev_A2_, Weight_, Bias_, z2_, a2_);

                        z2_.ToHost();
                        a2_.ToHost();
                        Sys.CudaSync();
                        prev_A2_.Free();
                        Weight_.Free();
                        Bias_.Free();
                        z2_.Free();
                        a2_.Free();
                    }
                }

                string key = string.Format("{0}x{1}", Z2.nRow, Z2.nCol);
                if (!SpanG.ContainsKey(key)) {
                    SpanG.Add(key, 0);
                    SpanC.Add(key, 0);
                    CntG.Add(key, 0);
                    CntC.Add(key, 0);
                }
                SpanG[key] += (DateTime.Now - start).TotalMilliseconds;
                CntG[key]++;

                if (Sys.GPUDebug) {

                    Array2 z2_sv = null;
                    Array2 a_sv = null;

                    start = DateTime.Now;

                    z2_sv = prev_A2.Dot(Weight) + Bias;
                    a_sv = z2_sv.Map(Sys.Sigmoid);

                    SpanC[key] += (DateTime.Now - start).TotalMilliseconds;
                    CntC[key]++;

                    double dz = (Z2 - z2_sv).Map(Math.Abs).Max();
                    double da = (Activation2 - a_sv).Map(Math.Abs).Max();
                    Debug.Assert(Math.Max(dz, da) < 0.000000001, "forward");

                    if (CntG[key] % 1000 == 0 || 1000 < Z2.nRow * Z2.nCol) {
                        Debug.WriteLine("time {0}x{1} GPU:{2} CPU:{3}", Z2.nRow, Z2.nCol, SpanG[key] / CntG[key], SpanC[key] / CntC[key]);
                    }
                }
            }
        }

        public override void Backward(Array2 Y) {
            if (NextLayer == null) {
                // 最後のレイヤーの場合

                dC_dA2 = Sys.CostDerivative(Activation2, Y);

                // cost = 1/2 * Σ xi*xi
                //Cost = xrange(dC_dA2.nCol).map(c => dC_dA2.Col(c).dt.map(x => x * x).reduce((x, y) => x + y)).map(x => x / 2);
                Cost = new Array1( from r in dC_dA2.Rows() select r.Map(x => x * x).Sum() / 2 );
            }
            else {
                // 最後のレイヤーでない場合

                FullyConnectedLayer next_layer = NextLayer as FullyConnectedLayer;

                // next.Zk = Σj next.Wkj * Aj
                // dC/dAi = Σk dC/d(next.Zk) * d(next.Zk)/dAi = Σk dC/d(next.Zk) * d(Σj next.Wkj * Aj)/dAi = Σk dC/d(next.Zk) * Wki
                dC_dA2 = next_layer.dC_dZ2.Dot(next_layer.Weight.T());
            }

            // dC/dZ = dC/dA * dA/dZ
            dC_dZ2 = dC_dA2 * Z2.Map(Sys.SigmoidPrime);

            if (Network.DoVerifyFull) {

                // Z = prev-A . Weight + Bias;
                // dC/dB = dC/dZ * dZ/dB = dC/dZ * 1 = dC/dZ
                NablaBiases = dC_dZ2;

                // dC/dWij = dC/dZ * dZ/dWij = dC/dZj * prev-Ai
                NablaWeights = new Array3(Network.MiniBatchSize, Weight.nRow, Weight.nCol);
                for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {
                    for (int r = 0; r < Weight.nRow; r++) {
                        for (int c = 0; c < Weight.nCol; c++) {
                            NablaWeights[batch_idx, r, c] = PrevLayer.GetActivation2()[batch_idx, r] * dC_dZ2[batch_idx, c];
                        }
                    }
                }
            }
        }

        public override void UpdateParameter() {
            Array1 NablaB = new Array1(from c in dC_dZ2.Cols() select c.Sum());
            Array2 NablaW = PrevLayer.GetActivation2().T().Dot( dC_dZ2 );

            Bias   = Bias   - Network.Eta2 * NablaB;
            Weight = Weight - Network.Eta2 * NablaW;
        }
    }

    public class Layer4 : Layer {
        public Array4 Activation4;

        public Array4 svActivation4;
        public Array4 svdC_dA4;
        public Array4 dC_dA4;
    }

    public class ConvolutionalLayer : Layer4 {
        public Array4 Z4;
        public Array4 svZ4;

        public int FilterSize;
        public int FilterCount;
        public Array1 Bias;
        public Array3 Weight3;
        public Array1 svBias;
        public Array3 svWeight3;

        public Array2 NablaBiases;
        public Array4 NablaWeight4;
        public Array2 svNablaBiases;
        public Array4 svNablaWeight4;

        public ConvolutionalLayer(int filter_size, int filter_count) : base() {
            FilterSize = filter_size;
            FilterCount = filter_count;
        }

        public override Array4 GetActivation4() {
            return Activation4;
        }

        public override void SaveParam() {
            svBias = Bias.Clone();
            svWeight3 = Weight3.Clone();

            svNablaBiases = NablaBiases.Clone();
            svNablaWeight4 = NablaWeight4.Clone();
        }

        public override void RestoreParam() {
            Bias = svBias.Clone();
            Weight3 = svWeight3.Clone();

            NablaBiases = svNablaBiases.Clone();
            NablaWeight4 = svNablaWeight4.Clone();
        }

        public override int ParamLength() {
            return Bias.Length + Weight3.Length;
        }

        public override double ParamAt(int i) {
            if (i < Bias.Length) {
                return Bias.dt[i];
            }

            int d, r, c;
            Weight3.DecomposeIdx(i - Bias.Length, out d, out r, out c);
            return Weight3.dt[d, r, c];
        }

        public override void ParamSet(int i, double p) {
            if (i < Bias.Length) {
                Bias.dt[i] = p;
                return;
            }

            int d, r, c;
            Weight3.DecomposeIdx(i - Bias.Length, out d, out r, out c);
            Weight3.dt[d, r, c] = p;
        }

        public override Array1 dC_dPs(int batch_idx) {
            int mini_batch_size = NablaBiases.nRow;
            Array1 v = new Array1( (NablaBiases.Length + NablaWeight4.Length) / mini_batch_size );

            int param_i = 0;
            for (int filter_idx = 0; filter_idx < Bias.Length; filter_idx++) {
                v.dt[param_i] = NablaBiases[batch_idx, filter_idx];

                param_i++;
            }

            // すべてのフィルターに対し
            for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                // フィルターの行に対し
                for (int r2 = 0; r2 < FilterSize; r2++) {

                    // フィルターの列に対し
                    for (int c2 = 0; c2 < FilterSize; c2++) {
                        v.dt[param_i] = NablaWeight4[batch_idx, filter_idx, r2, c2];

                        param_i++;
                    }
                }
            }

            return v;
        }

        public override Array1 dC_dPByParamIdx(int param_idx) {
            int mini_batch_size = NablaBiases.nRow;

            if (param_idx < Bias.Length) {

                return new Array1(from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaBiases[batch_idx, param_idx]);
            }

            int i = param_idx - Bias.Length;
            int filter_idx = i / (FilterSize * FilterSize);

            i -= filter_idx * (FilterSize * FilterSize);
            int r2 = i / FilterSize;
            int c2 = i - r2 * FilterSize;

            return new Array1(from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaWeight4[batch_idx, filter_idx, r2, c2]);
        }

        public override double dC_dPByParamIdxAvg(int param_idx) {
            int mini_batch_size = NablaBiases.nRow;

            if (param_idx < Bias.Length) {

                return (from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaBiases[batch_idx, param_idx]).Average();
            }

            int i = param_idx - Bias.Length;
            int filter_idx = i / (FilterSize * FilterSize);

            i -= filter_idx * (FilterSize * FilterSize);
            int r2 = i / FilterSize;
            int c2 = i - r2 * FilterSize;

            return (from batch_idx in Enumerable.Range(0, mini_batch_size) select NablaWeight4[batch_idx, filter_idx, r2, c2]).Average();
        }

        public override void init(Layer prev_layer) {
            base.init(prev_layer);

            Debug.Assert(PrevLayer is InputLayer, "Convolutional-Layer-init");

            ImgRows = PrevLayer.ImgRows - FilterSize + 1;
            ImgCols = PrevLayer.ImgCols - FilterSize + 1;
            UnitSize = ImgRows * ImgCols * FilterCount;

            Bias = TNormalRandom.randn(FilterCount);
            Weight3 = (new Array3(FilterCount, FilterSize, FilterSize)).Set(TNormalRandom.NormalRandom.NextDouble); 
        }

        public void ForwardCPU() {
            Array3 prev_activation = PrevLayer.GetActivation3();

            // バッチ内のデータに対し
            for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {

                // 出力の行に対し
                for (int r1 = 0; r1 < ImgRows; r1++) {

                    // 出力の列に対し
                    for (int c1 = 0; c1 < ImgCols; c1++) {

                        // すべてのフィルターに対し
                        for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                            double sum = 0.0;

                            // フィルターの行に対し
                            for (int r2 = 0; r2 < FilterSize; r2++) {

                                // フィルターの列に対し
                                for (int c2 = 0; c2 < FilterSize; c2++) {
                                    sum += prev_activation.dt[batch_idx, r1 + r2, c1 + c2] * Weight3.dt[filter_idx, r2, c2];
                                }
                            }

                            // 出力
                            double z_val = sum + Bias.dt[filter_idx];

                            Z4.dt[batch_idx, r1, c1, filter_idx] = z_val;
                            Activation4.dt[batch_idx, r1, c1, filter_idx] = Sys.Sigmoid(z_val);
                        }
                    }
                }
            }
        }

        public void ForwardGPU() {
            Array3 prev_A3 = PrevLayer.GetActivation3();

            unsafe{
                fixed (double* prev_A3_dev = prev_A3.dt, Weight3_dev = Weight3.dt, Bias_dev = Bias.dt, z4_dev = Z4.dt, a4_dev = Activation4.dt) {
                    Array3_ prev_A3_ = new Array3_(prev_A3_dev, prev_A3);
                    Array3_ Weight3_ = new Array3_(Weight3_dev, Weight3);
                    Array1_ Bias_ = new Array1_(Bias_dev, Bias);
                    Array4_ z4_ = new Array4_(z4_dev, Z4, false);
                    Array4_ a4_ = new Array4_(a4_dev, Activation4, false);

                    Sys.ConvolutionForward(prev_A3_, Weight3_, Bias_, z4_, a4_);

                    z4_.ToHost();
                    a4_.ToHost();
                    Sys.CudaSync();
                    prev_A3_.Free();
                    Weight3_.Free();
                    Bias_.Free();
                    z4_.Free();
                    a4_.Free();
                }
            }
        }

        static Dictionary<string, double> SpanC = new Dictionary<string, double>();
        static Dictionary<string, double> SpanG = new Dictionary<string, double>();
        static Dictionary<string, int> CntC = new Dictionary<string, int>();
        static Dictionary<string, int> CntG = new Dictionary<string, int>();
        double[] BGSpan = new double[4];
        int BGCnt = 0;

        public override void Forward() {
            Array3 prev_activation = PrevLayer.GetActivation3();

            if (Z4 == null || !Enumerable.SequenceEqual(Z4.Shape(), new int[] { Network.MiniBatchSize, ImgRows, ImgCols, FilterCount })) {

                Z4 = new Array4(Network.MiniBatchSize, ImgRows, ImgCols, FilterCount);
                Activation4 = new Array4(Network.MiniBatchSize, ImgRows, ImgCols, FilterCount);
            }


            if (Sys.CPU) {

                ForwardCPU();
            }
            else {

                string key = string.Format("{0}", prev_activation.nDepth);
                if (!SpanG.ContainsKey(key)) {
                    SpanG.Add(key, 0);
                    SpanC.Add(key, 0);
                    CntG.Add(key, 0);
                    CntC.Add(key, 0);
                }

                DateTime start = DateTime.Now;

                ForwardGPU();

                SpanG[key] += (DateTime.Now - start).TotalMilliseconds;
                CntG[key]++;

                if (Sys.GPUDebug) {

                    Array4 z4_sv = Z4.Clone();
                    Array4 a4_sv = Activation4.Clone();

                    start = DateTime.Now;

                    ForwardCPU();

                    SpanC[key] += (DateTime.Now - start).TotalMilliseconds;
                    CntC[key]++;

                    double dz = (Z4 - z4_sv).Map(Math.Abs).Max();
                    double da = (Activation4 - a4_sv).Map(Math.Abs).Max();
                    Debug.Assert(Math.Max(dz, da) < 0.000000001, "conv-forward");

                    if (CntG[key] % 100 == 0 || 100 < prev_activation.nDepth) {
                        Debug.WriteLine("time {0} GPU:{1} CPU:{2}", key, SpanG[key] / CntG[key], SpanC[key] / CntC[key]);
                    }
                }
            }
        }

        void NablaWeightCPU(Array3 prev_activation, Array4 deltaT) {
            // すべてのフィルターに対し
            for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                // フィルターの行に対し
                for (int r2 = 0; r2 < FilterSize; r2++) {

                    // フィルターの列に対し
                    for (int c2 = 0; c2 < FilterSize; c2++) {

                        // バッチ内のデータに対し
                        for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {

                            double nabla_w = 0.0;

                            // 出力の行に対し
                            for (int r1 = 0; r1 < ImgRows; r1++) {

                                // 出力の列に対し
                                for (int c1 = 0; c1 < ImgCols; c1++) {

                                    double delta = deltaT[batch_idx, r1, c1, filter_idx];
                                    if (delta != 0) {

                                        nabla_w += delta * prev_activation[batch_idx, r1 + r2, c1 + c2];
                                    }
                                }
                            }

                            NablaWeight4[batch_idx, filter_idx, r2, c2] = nabla_w;
                        }
                    }
                }
            }
        }

        void NablaWeightGPU(Array3 prev_A3, Array4 deltaT) {
            unsafe{
                fixed (double* prev_A3_dev = prev_A3.dt, deltaT_dev = deltaT.dt, NablaWeight4_dev = NablaWeight4.dt) {
                    Array3_ prev_A3_ = new Array3_(prev_A3_dev, prev_A3);
                    Array4_ deltaT_ = new Array4_(deltaT_dev, deltaT);
                    Array4_ NablaWeight4_ = new Array4_(NablaWeight4_dev, NablaWeight4, false);

                    Sys.ConvolutionNablaWeight(prev_A3_, deltaT_, NablaWeight4_);

                    NablaWeight4_.ToHost();
                    Sys.CudaSync();
                    prev_A3_.Free();
                    deltaT_.Free();
                    NablaWeight4_.Free();
                }
            }
        }

        public override void Backward(Array2 Y) {
            DateTime st = DateTime.Now;

            PoolingLayer next_layer = NextLayer as PoolingLayer;

            //dC_dZ4 = NextLayer.dC_dZ4.Mul(SigmoidPrime(Z4));
            Array4 deltaT = next_layer.dC_dZ4 * Z4.Map(Sys.SigmoidPrime);

            BGSpan[0] += (DateTime.Now - st).TotalMilliseconds;
            st = DateTime.Now;

            Array3 prev_activation = PrevLayer.GetActivation3();

            NablaBiases = new Array2(Network.MiniBatchSize, FilterCount);
            NablaWeight4 = new Array4(Network.MiniBatchSize, FilterCount, FilterSize, FilterSize);

            dC_dA4 = next_layer.dC_dZ4.Clone();

            BGSpan[1] += (DateTime.Now - st).TotalMilliseconds;
            st = DateTime.Now;

            // すべてのフィルターに対し
            for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                // バッチ内のデータに対し
                for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {

                    double nabla_b = 0.0;

                    // 出力の行に対し
                    for (int r1 = 0; r1 < ImgRows; r1++) {

                        // 出力の列に対し
                        for (int c1 = 0; c1 < ImgCols; c1++) {

                            nabla_b += deltaT[batch_idx, r1, c1, filter_idx];
                        }
                    }

                    NablaBiases[batch_idx, filter_idx] = nabla_b;
                }
            }
            BGSpan[2] += (DateTime.Now - st).TotalMilliseconds;
            st = DateTime.Now;

            if (Sys.CPU) {

                NablaWeightCPU(prev_activation, deltaT);
            }
            else {

                if (Sys.GPUDebug) {

                    NablaWeightCPU(prev_activation, deltaT);
                    Array4 NablaWeight4_sv = NablaWeight4.Clone();
                    NablaWeight4 = new Array4(NablaWeight4.Shape());

                    double d = (NablaWeight4 - NablaWeight4_sv).Map(Math.Abs).Max();
                    Debug.Assert(d < 0.000000001);
                }
                else {

                    NablaWeightGPU(prev_activation, deltaT);
                }
            }

            BGSpan[3] += (DateTime.Now - st).TotalMilliseconds;
            BGCnt++;

            if (BGCnt % 100 == 0) {

//                Debug.WriteLine("BG {0} {1} {2} {3}", BGSpan[0] / BGCnt, BGSpan[1] / BGCnt, BGSpan[2] / BGCnt, BGSpan[3] / BGCnt);
            }
        }

        public override void UpdateParameter() {
            // すべてのフィルターに対し
            //for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) { }
            Parallel.For(0, FilterCount, filter_idx => {
                double nabla_bias = (from batch_idx in Enumerable.Range(0, Network.MiniBatchSize) select NablaBiases[batch_idx, filter_idx]).Sum();
                Bias[filter_idx] -= Network.Eta2 * nabla_bias;

                // フィルターの行に対し
                for (int r2 = 0; r2 < FilterSize; r2++) {

                    // フィルターの列に対し
                    for (int c2 = 0; c2 < FilterSize; c2++) {
                        double nabla_w = 0;

                        // バッチ内のデータに対し
                        for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {
                            nabla_w += NablaWeight4[batch_idx, filter_idx, r2, c2];
                        }

                        Weight3[filter_idx, r2, c2] -= Network.Eta2 * nabla_w;
                    }
                }
            });
        }
    }


    public class PoolingLayer : Layer4 {
        public int FilterSize;
        public int FilterCount;
        public Array2 Activation2;
        public Array2 svActivation2;
        public int[,,,] MaxIdx;
        public bool RetainMaxIdx = false;
        public Array4 dC_dZ4;

        public PoolingLayer(int filter_size) : base() {
            FilterSize = filter_size;
        }

        public override Array4 GetActivation4() {
            return Activation4;
        }

        public override void init(Layer prev_layer) {
            base.init(prev_layer);

            Debug.Assert(PrevLayer is ConvolutionalLayer, "Pooling-Layer-init");

            ImgRows = PrevLayer.ImgRows / FilterSize;
            ImgCols = PrevLayer.ImgCols / FilterSize;
            FilterCount = (PrevLayer as ConvolutionalLayer).FilterCount;

            UnitSize = ImgRows * ImgCols * FilterCount;
        }

        public override Array2 GetActivation2() {
            return Activation2;
        }

        public override void Forward() {
            TW tw1 = new TW("pool-forward 1");

            ConvolutionalLayer prev_Layer = PrevLayer as ConvolutionalLayer;

            if (! RetainMaxIdx) {

                Activation4 = new Array4(Network.MiniBatchSize, ImgRows, ImgCols, FilterCount);
                MaxIdx = new int[Network.MiniBatchSize, ImgRows, ImgCols, FilterCount];
            }

            // バッチ内のデータに対し
            for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {

                // すべての行に対し
                for (int r1 = 0; r1 < ImgRows; r1++) {

                    // すべての列に対し
                    for (int c1 = 0; c1 < ImgCols; c1++) {

                        // すべてのフィルターに対し
                        for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                            if (RetainMaxIdx) {

                                int max_idx = MaxIdx[batch_idx, r1, c1, filter_idx];
                                int r2 = max_idx / FilterSize;
                                int c2 = max_idx - r2 * FilterSize;

                                Activation4[batch_idx, r1, c1, filter_idx] = prev_Layer.Activation4[batch_idx, r1 + r2, c1 + c2, filter_idx];
                            }
                            else {

                                double max_val = double.MinValue;
                                int max_idx = 0;

                                // フィルターの行に対し
                                for (int r2 = 0; r2 < FilterSize; r2++) {

                                    // フィルターの列に対し
                                    for (int c2 = 0; c2 < FilterSize; c2++) {

                                        double val = prev_Layer.Activation4[batch_idx, r1 + r2, c1 + c2, filter_idx];
                                        if (max_val < val) {

                                            max_val = val;
                                            max_idx = r2 * FilterSize + c2;
                                        }
                                    }
                                }

                                // 出力
                                Activation4[ batch_idx, r1, c1,  filter_idx] = max_val;
                                MaxIdx[ batch_idx, r1, c1,  filter_idx] = max_idx;
                            }
                        }
                    }
                }
            }

            tw1.Lap("pool-forward 2");
            Activation2 = (Array2)Activation4.Reshape(Network.MiniBatchSize, ImgRows * ImgCols * FilterCount);

            tw1.Stop();
        }

        public override void Backward(Array2 Y) {
            TW tw1 = new TW("pool-backward 1");

            ConvolutionalLayer prev_Layer = PrevLayer as ConvolutionalLayer;
            FullyConnectedLayer next_layer = NextLayer as FullyConnectedLayer;

            dC_dA4 = (Array4)next_layer.dC_dZ2.Dot(next_layer.Weight.T()).Reshape(Activation4.Shape());
            tw1.Lap("pool-backward 2");

            dC_dZ4 = new Array4(prev_Layer.Activation4.Shape());
            tw1.Lap("pool-backward 3");

            // バッチ内のデータに対し
            for (int batch_idx = 0; batch_idx < Network.MiniBatchSize; batch_idx++) {

                // すべての行に対し
                for (int r1 = 0; r1 < ImgRows; r1++) {

                    // すべての列に対し
                    for (int c1 = 0; c1 < ImgCols; c1++) {

                        // すべてのフィルターに対し
                        for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                            // 出力先
                            int max_idx = MaxIdx[batch_idx, r1, c1,  filter_idx];
                            int r2 = max_idx / FilterSize;
                            int c2 = max_idx - r2 * FilterSize;

                            dC_dZ4[batch_idx, r1 + r2, c1 + c2, filter_idx] = dC_dA4[batch_idx, r1, c1,  filter_idx];
                        }
                    }
                }
            }
            tw1.Stop();
        }
    }

    unsafe public struct Array1_ {
        public double* dt;
        public double* DevDt;
        public int Length;

        public Array1_(double* p, Array1 a, bool copy = true) {
            dt = p;
            Length = a.Length;

            DevDt = Sys.CudaAlloc(Length);
            if (copy) {
                Sys.CudaToDev(DevDt, dt, Length);
            }
        }

        public void ToHost() {
            Sys.CudaToHost(DevDt, dt, Length);
        }

        public void Free() {
            Sys.CudaFree(DevDt);
        }
    }

    unsafe public struct Array2_ {
        public double* dt;
        public double* DevDt;
        public int Length;
        public int nRow;
        public int nCol;

        public Array2_(double* p, Array2 a, bool copy = true) {
            dt = p;
            nRow = a.nRow;
            nCol = a.nCol;
            Length = nRow * nCol;

            DevDt = Sys.CudaAlloc(Length);
            if (copy) {
                Sys.CudaToDev(DevDt, dt, Length);
            }
        }

        public void ToHost() {
            Sys.CudaToHost(DevDt, dt, Length);
        }

        public void Free() {
            Sys.CudaFree(DevDt);
        }
    }

    unsafe public struct Array3_ {
        public double* dt;
        public double* DevDt;
        public int Length;
        public int nDepth;
        public int nRow;
        public int nCol;
        public int nRowCol;

        public Array3_(double* p, Array3 a, bool copy = true) {
            dt = p;
            nDepth = a.nDepth;
            nRow = a.nRow;
            nCol = a.nCol;
            nRowCol = nRow * nCol;

            Length = nDepth * nRow * nCol;

            DevDt = Sys.CudaAlloc(Length);
            if (copy) {
                Sys.CudaToDev(DevDt, dt, Length);
            }
        }

        public void ToHost() {
            Sys.CudaToHost(DevDt, dt, Length);
        }

        public void Free() {
            Sys.CudaFree(DevDt);
        }
    }

    unsafe public struct Array4_ {
        public double* dt;
        public double* DevDt;
        public int Length;
        public fixed int Dims[4];
        public fixed int Sizes[3];

        public Array4_(double* p, Array4 a, bool copy = true) {
            dt = p;
            fixed (int* d = Dims, s = Sizes) {
                d[0] = a.dt.GetLength(0);
                d[1] = a.dt.GetLength(1);
                d[2] = a.dt.GetLength(2);
                d[3] = a.dt.GetLength(3);

                s[2] = d[3];
                s[1] = d[3] * d[2];
                s[0] = d[3] * d[2] * d[1];

                Length = d[3] * d[2] * d[1] * d[0];
            }

            DevDt = Sys.CudaAlloc(Length);
            if (copy) {
                Sys.CudaToDev(DevDt, dt, Length);
            }
        }

        public void ToHost() {
            Sys.CudaToHost(DevDt, dt, Length);
        }

        public void Free() {
            Sys.CudaFree(DevDt);
        }
    }

    public partial class Network {
        public Layer[] Layers;

        public byte[,] TrainImage;
        public byte[] TrainLabel;
        public byte[,] TestImage;
        public byte[] TestLabel;

        public static void TestCUDA() {
            unsafe{
                Sys.CudaSetDevice(0);

                {
                    Array1 a = new Array1(new double[] { 1, 2, 3, 4, 5 });
                    Array1 b = new Array1(new double[] { 10, 20, 30, 40, 50 });
                    Array1 c = new Array1(new double[a.Length]);

                    fixed (double* adev = a.dt) {
                        fixed (double* bdev = b.dt) {
                            fixed (double* cdev = c.dt) {
                                Array1_ ma = new Array1_(adev, a);
                                Array1_ mb = new Array1_(bdev, b);
                                Array1_ mc = new Array1_(cdev, c, false);

                                Sys.CudaAdd1(ma, mb, mc);

                                mc.ToHost();
                                Sys.CudaSync();
                                ma.Free();
                                mb.Free();
                                mc.Free();
                            }
                        }

                    }
                    Debug.WriteLine("c:" + c.ToString());
                }
                {
                    Array2 a = new Array2(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
                    Array2 b = new Array2(new double[,] { { 10, 20, 30 }, { 40, 50, 60 } });
                    Array2 c = new Array2(new double[a.nRow, a.nCol]);

                    fixed (double* adev = a.dt, bdev = b.dt, cdev = c.dt) {
                        Array2_ ma = new Array2_(adev, a);
                        Array2_ mb = new Array2_(bdev, b);
                        Array2_ mc = new Array2_(cdev, c, false);

                        Sys.CudaAdd2(ma, mb, mc);

                        mc.ToHost();
                        Sys.CudaSync();
                        ma.Free();
                        mb.Free();
                        mc.Free();
                    }
                    Debug.WriteLine("a + b:" + (a + b).ToString());
                    Debug.WriteLine("c    :" + c.ToString());
                }
                {
                    Array3 a = new Array3(new double[2, 3, 4]);
                    Array3 b = new Array3(new double[2, 3, 4]);
                    Array3 c = new Array3(a.nDepth, a.nRow, a.nCol);
                    int n = 0;
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 3; j++) {
                            for (int k = 0; k < 4; k++) {
                                a[i, j, k] = n;
                                b[i, j, k] = n * 100;
                                n++;
                            }
                        }
                    }

                    fixed (double* adev = a.dt, bdev = b.dt, cdev = c.dt) {
                        Array3_ ma = new Array3_(adev, a);
                        Array3_ mb = new Array3_(bdev, b);
                        Array3_ mc = new Array3_(cdev, c, false);

                        Sys.CudaAdd3(ma, mb, mc);

                        mc.ToHost();
                        Sys.CudaSync();
                        ma.Free();
                        mb.Free();
                        mc.Free();
                    }
                    Debug.WriteLine("a + b:" + (a + b).ToString());
                    Debug.WriteLine("c    :" + c.ToString());
                }
                {
                    Array4 a = new Array4(new double[2, 3, 4, 5]);
                    Array4 b = new Array4(new double[2, 3, 4, 5]);
                    Array4 c = new Array4(a.Shape());
                    int n = 0;
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 3; j++) {
                            for (int k = 0; k < 4; k++) {
                                for (int l = 0; l < 5; l++) {
                                    a[i, j, k, l] = n;
                                    b[i, j, k, l] = n * 100;
                                    n++;
                                }
                            }
                        }
                    }
                    fixed (double* adev = a.dt, bdev = b.dt, cdev = c.dt) {
                        Array4_ ma = new Array4_(adev, a);
                        Array4_ mb = new Array4_(bdev, b);
                        Array4_ mc = new Array4_(cdev, c, false);

                        Sys.CudaAdd4(ma, mb, mc);

                        mc.ToHost();
                        Sys.CudaSync();
                        ma.Free();
                        mb.Free();
                        mc.Free();
                    }
                    Debug.WriteLine("a + b:" + (a + b).ToString());
                    Debug.WriteLine("c    :" + c.ToString());
                }
                {
                    Array2 a = new Array2(new double[,] { { 0.1, 0.2 }, { 0.3, 0.4 }, { 0.5, 0.6 } });
                    Array2 b = new Array2(new double[,] { { 0.11, 0.22, 0.33 }, { 0.44, 0.55, 0.66 } });
                    Array2 c = new Array2(new double[a.nRow, b.nCol]);
                    Array2 d = new Array2(new double[a.nRow, b.nCol]);

                    fixed (double* adev = a.dt, bdev = b.dt, cdev = c.dt, ddev = d.dt) {
                        Array2_ ma = new Array2_(adev, a);
                        Array2_ mb = new Array2_(bdev, b);
                        Array2_ mc = new Array2_(cdev, c, false);
                        Array2_ md = new Array2_(ddev, d, false);

                        Sys.CudaDotSigmoid(ma, mb, mc, md);

                        mc.ToHost();
                        md.ToHost();
                        Sys.CudaSync();
                        ma.Free();
                        mb.Free();
                        mc.Free();
                        md.Free();
                    }
                    Debug.WriteLine("c  :" + c.ToString());
                    Debug.WriteLine("a.b:" + a.Dot(b).ToString());
                    Debug.WriteLine("d  :" + d.ToString());
                    Debug.WriteLine("Sigmoid(a.b):" + a.Dot(b).Map(Sys.Sigmoid).ToString());
                }

//                Sys.CudaDeviceReset();
            }
        }

        public Network(Layer[] layers) {
            Layers = layers;

            Layer prev_layer = null;
            foreach(Layer layer in layers) {
                layer.init(prev_layer);
                prev_layer = layer;
            }
        }

        public FullyConnectedLayer LastLayer {
            get { return Layers[Layers.Length - 1] as FullyConnectedLayer; }
        }

        public InputLayer FirstLayer {
            get { return Layers[0] as InputLayer; }
        }

        public void SGD() {
            Network.TestCUDA();

            int train_cnt = TrainImage.GetLength(0);
            int data_len = TrainImage.GetLength(1);
            int mini_batch_cnt = train_cnt / MiniBatchSize;

            LastActivation = new Array3(Epochs, mini_batch_cnt, 10);
            DiffA = new Array3(mini_batch_cnt, MiniBatchSize, 10);

            for (int epoch_idx = 0; epoch_idx < Epochs; epoch_idx++) {
                Sys.CudaSetDevice(0);

                //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                //int e1 = Evaluate();
                //Debug.WriteLine("Epoch {0}: {1} / {2}", epoch_idx, e1, TestImage.GetLength(0));
                //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                int[] idxes = Sys.RandomSampling(train_cnt, train_cnt);

                for (int mini_batch_idx = 0; mini_batch_idx < mini_batch_cnt; mini_batch_idx++ ) {

                    Array2 X = new Array2(MiniBatchSize, data_len);
                    Array2 Y = new Array2(MiniBatchSize, 10);

                    for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                        var idx = idxes[mini_batch_idx * MiniBatchSize + batch_idx ];
                        Y[batch_idx, TrainLabel[idx]] = 1;

                        for(int k = 0; k < data_len; k++) {
                            X[batch_idx, k] = TrainImage[idx, k] / 256.0;
                        }
                    }

                    UpdateMiniBatch(X, Y, epoch_idx, idxes, mini_batch_cnt, mini_batch_idx);

                    if(mini_batch_idx % 100 == 0) {
                        if (mini_batch_idx % 1000 == 0) {

                            //TW.Stats();
                        }
                        Debug.WriteLine("Epoch {0}: idx:{1} cost:{2}", epoch_idx, mini_batch_idx, LastLayer.Cost.Avg());
                    }
                }

                int e = Evaluate();
                Debug.WriteLine("Epoch {0}: {1} / {2}", epoch_idx, e, TestImage.GetLength(0));

                Sys.CudaDeviceReset();
            }
        }

        void UpdateMiniBatch(Array2 X, Array2 Y, int epoch_idx, int[] idxes, int mini_batch_cnt, int mini_batch_idx) {
            TW tw1 = new TW("update-mini-batch forward");
            if(FirstLayer.NextLayer is FullyConnectedLayer) {

                FirstLayer.Activation2 = X;
            }
            else {

                FirstLayer.Activation3 = (Array3)X.Reshape(MiniBatchSize, FirstLayer.ImgRows, FirstLayer.ImgCols);
            }

            foreach (Layer layer in Layers) {
                layer.forward2();
            }
            tw1.Lap("update-mini-batch back");

            for (int i = Layers.Length - 1; 1 <= i; i--) {
                Layers[i].backward2(Y);
            }
            tw1.Lap("update-mini-batch param");

            if (DoVerifySingleParam || DoVerifyMultiParam || DoVerifyMultiParamAll || DoVerifyDeltaActivation2 || DoVerifyDeltaActivation4) {

                Verify(X, Y);
            }

            if (DoVerifyUpdateParameter) {

                VerifyUpdateParameter(X, Y, epoch_idx, mini_batch_cnt, mini_batch_idx);
            }
            else {

                foreach (Layer layer in Layers) {
                    layer.UpdateParameter();
                }
            }
            tw1.Stop();
        }

        int Evaluate() {
            int e_sum = 0;
            int test_cnt = 500;
            int svMiniBatchSize = MiniBatchSize;
            MiniBatchSize = test_cnt;

            int data_len = TestImage.GetLength(1);

            Array2 X = new Array2(test_cnt, data_len);
            int[] label = new int[test_cnt];
            StringWriter sw = new StringWriter();

            for(int test_idx = 0; test_idx < TestImage.GetLength(0) / test_cnt; test_idx++) {

                int offset = test_idx * test_cnt;

                for (int batch_idx = 0; batch_idx < test_cnt; batch_idx++) {
                    for (int k = 0; k < data_len; k++) {
                        X[batch_idx, k] = TestImage[offset + batch_idx, k] / 256.0;
                    }
                    label[batch_idx] = TestLabel[offset + batch_idx];
                }

                if (FirstLayer.NextLayer is FullyConnectedLayer) {

                    FirstLayer.Activation2 = X;
                }
                else {

                    FirstLayer.Activation3 = (Array3)X.Reshape(test_cnt, FirstLayer.ImgRows, FirstLayer.ImgCols);
                }

                foreach (Layer layer in Layers) {
                    layer.Forward();
                }

                Array2 result = LastLayer.GetActivation2();

                //Debug.WriteLine("テスト --------------------------------------------------------------------------------");
                //Debug.WriteLine(result.ToString());

                File.WriteAllText("LastA.csv", result.ToString());
                Array1 a = new Array1(from r in Enumerable.Range(0, result.nRow) select (double)result.Row(r).ArgMax());
                Debug.Assert(a.Length == label.Length);
                for (int i = 0; i < a.Length; i++) {
                    sw.WriteLine("{0},{1}", a.dt[i], label[i]);
                }

                int e = (int)(from r in Enumerable.Range(0, result.nRow) select result.Row(r).ArgMax() == label[r] ? 1 : 0).Sum();
                e_sum += e;
            }

            File.WriteAllText("ArgMax.csv", sw.ToString());

            MiniBatchSize = svMiniBatchSize;

            return e_sum;
        }
    }

    public class Sys {
        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaSetDevice(int device);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaDeviceReset();

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static double* CudaAlloc(int len);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaFree(double* dev);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaToDev(double* dev, double* dt, int len);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaToHost(double* dev, double* dt, int len);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaSync();

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaAdd1(Array1_ a, Array1_ b, Array1_ c);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaAdd2(Array2_ a, Array2_ b, Array2_ c);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaAdd3(Array3_ a, Array3_ b, Array3_ c);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaAdd4(Array4_ a, Array4_ b, Array4_ c);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int CudaDotSigmoid(Array2_ a, Array2_ b, Array2_ c, Array2_ d);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int FullForward(Array2_ prev_A2_, Array2_ Weight_, Array1_ Bias_, Array2_ z2_, Array2_ a2_);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int ConvolutionForward(Array3_ prev_A3, Array3_ weight3, Array1_ bias, Array4_ z4, Array4_ a4);

        [DllImport("CUDALib.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe public extern static int ConvolutionNablaWeight(Array3_ prev_A3, Array4_ deltaT, Array4_ NablaWeight4);

        public static bool isFloat64 = true;
        public static bool DebugOut = true;

        public static bool GPUDebug = false;
        public static bool isCNN = true;
        public static bool CPU = false;

        public static double Sigmoid(double z){
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public static double SigmoidPrime(double z) {
            double f = Sigmoid(z);
            return f * (1 - f);
        }

        public static Array2 CostDerivative(Array2 output_activations, Array2 y){
            return output_activations - y;
        }


        public static int[] RandomSampling(int all_count, int sample_count) {
            int[] ret = new int[sample_count];

            int[] numbers = new int[all_count];
            for (int i = 0; i < all_count; i++) {
                numbers[i] = i;
            }

            for (int i = 0; i < sample_count; i++) {
                int n = TNormalRandom.Rn.Next(all_count - i);

                ret[i] = numbers[n];
                numbers[n] = numbers[all_count - i - 1];
            }
            /*
            for (int i = 0; i < sample_count; i++) {
                for (int j = i + 1; j < sample_count; j++) {
                    Debug.Assert(ret[i] != ret[j]);
                }
            }
             */

            return ret;
        }

        public static void TestRandomSampling() {
            for (int i = 0; i < 100; i++) {
                int[] v = RandomSampling(20, 10);
                for (int k = 0; k < v.Length; k++) {
                    Debug.Write(" " + v[k].ToString());
                }
                Debug.WriteLine("");
            }

        }
    }
}
