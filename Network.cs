﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace MachineLearning {
    public class Layer {
        public Network ParentNetwork;
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

        public virtual void Forward() {
        }

        public virtual void Backward(Array2 Y, double eta2) {
        }

        public void forward2() {
            DateTime startTime = DateTime.Now;
            Forward();
            FwCnt++;
            FwTime += (DateTime.Now - startTime).TotalMilliseconds;
        }

        public void backward2(Array2 Y, double eta2) {
            DateTime startTime = DateTime.Now;
            Backward(Y, eta2);
            BwCnt++;
            BwTime += (DateTime.Now - startTime).TotalMilliseconds;
        }

        public virtual void UpdateParameter(double eta2) {
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
        public Array2 CostDerivative2;
        public Array1 Cost;

        public Array2 svActivation2;
        public Array2 svZ2;
        public Array2 svCostDerivative2;

        public Array1 Bias;
        public Array2 Weight;

        public Array1 NablaB;
        public Array2 NablaW;

        public Array2 Delta;
        public Array2 NablaBiases;
        public Array3 NablaWeights;

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

        public override void Forward() {
            Z2 = PrevLayer.GetActivation2().Dot(Weight) + Bias;
            Activation2 = Z2.Map(Sys.Sigmoid);
        }

        public override void Backward(Array2 Y, double eta2) {
            if (NextLayer == null) {
                // 最後のレイヤーの場合

                CostDerivative2 = Sys.CostDerivative(Activation2, Y);

                // cost = 1/2 * Σ xi*xi
                //Cost = xrange(CostDerivative2.nCol).map(c => CostDerivative2.Col(c).dt.map(x => x * x).reduce((x, y) => x + y)).map(x => x / 2);
                Cost = new Array1( from r in CostDerivative2.Rows() select r.Map(x => x * x).Sum() / 2 );
            }
            else {
                // 最後のレイヤーでない場合

                FullyConnectedLayer next_layer = NextLayer as FullyConnectedLayer;

                // next.(dC/dA)
                // dC/dAj = Σ dC/d(next.Zk) * d(next.Zk)/dAj = Σ dC/d(next.Zk) * Wk
                // d(next.Z)/dA = d(Σ Wk * Ak)/dA = Σ Wk * d(Ak)/dA

                // next_layer.Delta = dC/d(next.Z) = dC/d(Σ next.Wk * Ak)
                // next.Zk = Σ next.Wki * Ai

                // dC/dAi = Σ dC/d(next.Zk) * d(next.Zk)/dAi = 

                CostDerivative2 = next_layer.Delta.Dot(next_layer.Weight.T());
            }

            // dC/dZ = dC/dA * dA/dZ
            this.Delta = CostDerivative2 * Z2.Map(Sys.SigmoidPrime);

            NablaB = new Array1(from c in Delta.Cols() select c.Sum());
            NablaW = PrevLayer.GetActivation2().T().Dot( this.Delta );

            if (Sys.isDebug) {

                NablaBiases = this.Delta;
                // constructor(rows, cols, init, column_major, depth)
                NablaWeights = new Array3(ParentNetwork.MiniBatchSize, Weight.nRow, Weight.nCol);
                for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {
                    for (int r = 0; r < Weight.nRow; r++) {
                        for (int c = 0; c < Weight.nCol; c++) {
                            NablaWeights[batch_idx, r, c] = PrevLayer.GetActivation2()[batch_idx, r] * this.Delta[batch_idx, c];
                        }
                    }
                }
            }
        }

        public override void UpdateParameter(double eta2) {
            Weight = Weight - eta2 * NablaW;
            Bias = Bias - eta2 * NablaB;
        }
    }

    public class Layer4 : Layer {
        public Array4 Activation4;

        public Array4 svActivation4;
        public Array4 svCostDerivative4;
        public Array4 CostDerivative4;
    }

    public class ConvolutionalLayer : Layer4 {
        public Array4 Z4;
        public Array4 svZ4;

        public int FilterSize;
        public int FilterCount;
        public Array1 Bias;
        public Array3 Weight3;
        public Array2 NablaBiases;
        public Array4 NablaWeight4;

        public ConvolutionalLayer(int filter_size, int filter_count) : base() {
            FilterSize = filter_size;
            FilterCount = filter_count;
        }

        public override Array4 GetActivation4() {
            return Activation4;
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

        public override void Forward() {
            if (Z4 == null || !Enumerable.SequenceEqual(Z4.Shape(), new int[] { ParentNetwork.MiniBatchSize, ImgRows, ImgCols, FilterCount }) ) {

                Z4 = new Array4(ParentNetwork.MiniBatchSize, ImgRows, ImgCols, FilterCount);
                Activation4 = new Array4(ParentNetwork.MiniBatchSize, ImgRows, ImgCols, FilterCount);
            }

            Array3 prev_activation = PrevLayer.GetActivation3();

            // バッチ内のデータに対し
            for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {

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

        public override void Backward(Array2 Y, double eta2) {
            PoolingLayer next_layer = NextLayer as PoolingLayer;

            //this.Delta = NextLayer.Delta.Mul(SigmoidPrime(Z4));
            Array4 deltaT = next_layer.Delta * Z4.Map(Sys.SigmoidPrime);

            Array3 prev_activation = PrevLayer.GetActivation3();

            NablaBiases = new Array2(ParentNetwork.MiniBatchSize, FilterCount);
            NablaWeight4 = new Array4(ParentNetwork.MiniBatchSize, FilterCount, FilterSize, FilterSize);
            CostDerivative4 = new Array4(ParentNetwork.MiniBatchSize, ImgRows, ImgCols, FilterCount);

            // すべてのフィルターに対し
            for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                // バッチ内のデータに対し
                for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {

                    double nabla_b = 0.0;

                    // 出力の行に対し
                    for (int r1 = 0; r1 < ImgRows; r1++) {

                        // 出力の列に対し
                        for (int c1 = 0; c1 < ImgCols; c1++) {

                            nabla_b += deltaT[batch_idx, r1, c1, filter_idx];
                            CostDerivative4[batch_idx, r1, c1, filter_idx] = next_layer.Delta[batch_idx, r1, c1, filter_idx];
                        }
                    }

                    NablaBiases[batch_idx, filter_idx] = nabla_b;
                }
            }

            // すべてのフィルターに対し
            for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                // フィルターの行に対し
                for (int r2 = 0; r2 < FilterSize; r2++) {

                    // フィルターの列に対し
                    for (int c2 = 0; c2 < FilterSize; c2++) {

                        // バッチ内のデータに対し
                        for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {

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

        public override void UpdateParameter(double eta2) {
            double eta3 = eta2 / (FilterSize * FilterSize);

            // すべてのフィルターに対し
            for (int filter_idx = 0; filter_idx < FilterCount; filter_idx++) {

                double nabla_bias = (from batch_idx in Enumerable.Range(0, ParentNetwork.MiniBatchSize) select NablaBiases[batch_idx, filter_idx]).Sum();
                Bias[filter_idx] -= eta3 * nabla_bias;

                // フィルターの行に対し
                for (int r2 = 0; r2 < FilterSize; r2++) {

                    // フィルターの列に対し
                    for (int c2 = 0; c2 < FilterSize; c2++) {
                        double nabla_w = 0;

                        // バッチ内のデータに対し
                        for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {
                            nabla_w += NablaWeight4[batch_idx, filter_idx, r2, c2];
                        }

                        Weight3[filter_idx, r2, c2] -= eta3 * nabla_w;
                    }
                }
            }
        }
    }


    public class PoolingLayer : Layer4 {
        public int FilterSize;
        public int FilterCount;
        public Array2 Activation2;
        public Array2 svActivation2;
        public int[,,,] MaxIdx;
        public bool RetainMaxIdx = false;
        public Array4 Delta;

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
            ConvolutionalLayer prev_Layer = PrevLayer as ConvolutionalLayer;
            Activation4 = new Array4(ParentNetwork.MiniBatchSize, ImgRows, ImgCols, FilterCount);

            MaxIdx = new int[ParentNetwork.MiniBatchSize, ImgRows, ImgCols, FilterCount];

            // バッチ内のデータに対し
            for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {

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

            Activation2 = (Array2)Activation4.Reshape(ParentNetwork.MiniBatchSize, ImgRows * ImgCols * FilterCount);
        }

        public override void Backward(Array2 Y, double eta2) {
            ConvolutionalLayer prev_Layer = PrevLayer as ConvolutionalLayer;
            FullyConnectedLayer next_layer = NextLayer as FullyConnectedLayer;

            CostDerivative4 = (Array4)next_layer.Delta.Dot(next_layer.Weight.T()).Reshape(Activation4.Shape());

            Delta = new Array4(prev_Layer.Activation4.Shape());

            // バッチ内のデータに対し
            for (int batch_idx = 0; batch_idx < ParentNetwork.MiniBatchSize; batch_idx++) {

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

                            Delta[batch_idx, r1 + r2, c1 + c2, filter_idx] = CostDerivative4[batch_idx, r1, c1,  filter_idx];
                        }
                    }
                }
            }
        }
    }


    public partial class Network {
        public int MiniBatchSize;
        public Layer[] Layers;

        public byte[,] TrainImage;
        public byte[] TrainLabel;
        public byte[,] TestImage;
        public byte[] TestLabel;

        public Network(Layer[] layers) {
            Layers = layers;

            Layer prev_layer = null;
            foreach(Layer layer in layers) {
                layer.ParentNetwork = this;
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

        public void SGD(int epochs, int mini_batch_size, double eta) {
            MiniBatchSize = mini_batch_size;

            int train_cnt = TrainImage.GetLength(0);
            int data_len = TrainImage.GetLength(1);
            for (int j = 0; j < epochs; j++) {

                int[] idxes = Sys.RandomSampling(train_cnt, train_cnt);

                int mini_batch_cnt = train_cnt / MiniBatchSize;
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

                    UpdateMiniBatch(X, Y, eta);

                    if(mini_batch_idx % 1000 == 0) {
                        Debug.WriteLine("mini batch idx {0}", mini_batch_idx);
                    }
                }

                int e = Evaluate();
                Debug.WriteLine("Epoch {0}: {1} / {2}", j, e, TestImage.GetLength(0));
            }
        }

        void UpdateMiniBatch(Array2 X, Array2 Y, double eta) {
            if(FirstLayer.NextLayer is FullyConnectedLayer) {

                FirstLayer.Activation2 = X;
            }
            else {

                FirstLayer.Activation3 = (Array3)X.Reshape(MiniBatchSize, FirstLayer.ImgRows, FirstLayer.ImgCols);
            }

            foreach (Layer layer in Layers) {
                layer.forward2();
            }

            double eta2 = eta / MiniBatchSize;

            for (int i = Layers.Length - 1; 1 <= i; i--) {
                Layers[i].backward2(Y, eta2);
            }

            if (Sys.isDebug) {

                Verify(X, Y, eta2);
            }

            foreach (Layer layer in Layers) {
                layer.UpdateParameter(eta2);
            }
        }

        int Evaluate() {
            int test_cnt = TestImage.GetLength(0);
            int data_len = TestImage.GetLength(1);

            Array2 X = new Array2(test_cnt, data_len);

            for (int batch_idx = 0; batch_idx < test_cnt; batch_idx++) {
                for (int k = 0; k < data_len; k++) {
                    X[batch_idx, k] = TestImage[batch_idx, k] / 256.0;
                }
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

            Debug.WriteLine("テスト --------------------------------------------------------------------------------");
            Debug.WriteLine(result.ToString());

            return (from r in Enumerable.Range(0, result.nRow) select result.Row(r).ArgMax() == TestLabel[r] ? 1 : 0).Sum();
        }
    }

    public class Sys {
        public static bool isFloat64 = true;// isDebug;
        public static bool DebugOut = true;

        public static bool isDebug = false;
        public static bool isCNN = false;

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
