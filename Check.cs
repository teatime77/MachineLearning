using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;

namespace MachineLearning {
    public partial class Network {
        bool DoVerifySub2 = false;
        bool DoVerifySub4 = true;
        bool DoVerifyDeltaActivation2 = false;
        bool DoVerifyDeltaActivation4 = false;

        Array3 LastActivation;
        Array3 DiffA;

        void Output(string path, Array3 ret) {
            double err = ret.Map(Math.Abs).Max();
            double avg = ret.Map(Math.Abs).Sum() / ret.dt.Length;

            File.WriteAllText(path + ".csv", ret.ToString());
        }

        void VerifySub2(Array2 X, Array2 Y, Array1 sv_cost, FullyConnectedLayer layer, double delta_param, Array1 nabla, double[,,] ret, int i_ret) {
            for (Layer l = layer; l != null; l = l.NextLayer) {
                l.forward2();
            }

            for (Layer l = LastLayer; ; l = l.PrevLayer) {
                l.backward2(Y);
                if (l == layer) {
                    break;
                }
            }

            Array2 dActivation2 = layer.Activation2 - layer.svActivation2;
            Array2 dZ = layer.Z2 - layer.svZ2;
            Array1 dActivation2_dC_dA = (dActivation2 * layer.dC_dA2).SumRow();
            Array1 dCost = LastLayer.Cost - sv_cost;

            // ΔC ≒ Δparam * nabla 
            Array1 delta_param_nabla = delta_param * nabla;
            CheckEqual1(delta_param_nabla, dCost, out ret[i_ret, 0, 0], out ret[i_ret, 0, 1]);

            // ΔC ≒ ΔA * dC/dA
            CheckEqual1(dActivation2_dC_dA, dCost, out ret[i_ret, 1, 0], out ret[i_ret, 1, 1]);

            Array2 dA_dZ = layer.svZ2.Map(Sys.SigmoidPrime);
            Array2 dZ_dA_dZ = dZ * dA_dZ;

            // ΔA ≒ ΔZ * dA/dZ
            CheckEqual2(dZ_dA_dZ, dActivation2, out ret[i_ret, 2, 0], out ret[i_ret, 2, 1]);
        }

        void VerifySub4(Array2 X, Array2 Y, Array1 sv_cost, ConvolutionalLayer layer, int filter_idx, int r2, int c2, double delta_param, Array1 nabla, double[,,] ret, int i_ret) {
            for (Layer l = layer; l != null; l = l.NextLayer) {
                l.forward2();
            }

            for (Layer l = LastLayer; ; l = l.PrevLayer) {
                l.backward2(Y);
                if (l == layer) {
                    break;
                }
            }

            Array4 dZ = layer.Z4 - layer.svZ4;
            Array4 dA = layer.Activation4 - layer.svActivation4;

            int row_size = dA.dt.GetLength(0);
            int col_size = dA.dt.GetLength(1) * dA.dt.GetLength(2) * dA.dt.GetLength(3);

            Array1 dA_dC_dA = ((dA * layer.dC_dA4).Reshape(row_size, col_size) as Array2).SumRow();
            Array1 dC = LastLayer.Cost - sv_cost;

            // ΔC ≒ Δparam * nabla 
            Array1 delta_param_nabla = delta_param * nabla;
            CheckEqual1(dC, delta_param_nabla, out ret[i_ret, 0, 0], out ret[i_ret, 0, 1]);

            // ΔC ≒ ΔA * dC/dA
            CheckEqual1(dC, dA_dC_dA, out ret[i_ret, 1, 0], out ret[i_ret, 1, 1]);

            Array4 dA_dZ = layer.svZ4.Map(Sys.SigmoidPrime);
            Array4 dZ_dA_dZ = dZ * dA_dZ;

            // ΔA ≒ ΔZ * dA/dZ
            CheckEqual2(dA.Reshape(row_size, col_size) as Array2, dZ_dA_dZ.Reshape(row_size, col_size) as Array2, out ret[i_ret, 2, 0], out ret[i_ret, 2, 1]);
        }


        void VerifyDeltaActivation2(Array2 X, Array2 Y, Array1 sv_cost, FullyConnectedLayer layer) {
            Array2 A = layer.GetActivation2();

            Array2 Err2 = new Array2(A.nRow, A.nCol);
            Array2 Err3 = new Array2(A.nRow, A.nCol);

            for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                for (int k = 0; k < A.nCol; k++) {
                    double delta_z = 0;
                    double delta_a;

                    double a_sv = A[batch_idx, k];
                    double z_sv = 0;
                    double cost_deriv = layer.svdC_dA2[batch_idx, k];

                    z_sv = layer.Z2.dt[batch_idx, k];
                    delta_z = z_sv * 0.001;
                    layer.Z2.dt[batch_idx, k] += delta_z;

                    A[batch_idx, k] = Sys.Sigmoid(layer.Z2.dt[batch_idx, k]);
                    delta_a = A[batch_idx, k] - a_sv;

                    for (Layer L = layer.NextLayer; L != null; L = L.NextLayer) {
                        L.forward2();
                    }

                    for (Layer L = LastLayer; L != null; L = L.PrevLayer) {
                        L.backward2(Y);
                        if (L == layer) {
                            break;
                        }
                    }

                    //-------------------- ΔC
                    double deltaC = LastLayer.Cost[batch_idx] - sv_cost[batch_idx];

                    //-------------------- ΔC ≒ Δa0 * δC/δa0
                    double deltaC2 = delta_a * cost_deriv;

                    Err2[batch_idx, k] = Math.Abs(deltaC2 - deltaC);

                    double dA_dZ = Sys.SigmoidPrime(z_sv);

                    //ΔC ≒ Δz0 * δC/δa0 * da0/dz0
                    double deltaC3 = delta_z * cost_deriv * dA_dZ;

                    Err3[batch_idx, k] = Math.Abs(deltaC3 - deltaC);

                    layer.Z2.dt[batch_idx, k] = z_sv;

                    A[batch_idx, k] = a_sv;
                }
            }

            double max_err2 = Err2.Max();
            double max_err3 = Err3.Max();

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.Activation2 = player.svActivation2.Clone();
            }
        }

        void VerifyDeltaActivation4(Array2 X, Array2 Y, Array1 sv_cost, Layer4 layer) {
            Array4 A = layer.GetActivation4();
            int n1 = A.dt.GetLength(1);
            int n2 = A.dt.GetLength(2);
            int n3 = A.dt.GetLength(3);

            Array3 ret = new Array3(n1 * n2 * n3, 2, 2);
            int i_ret = 0;

            for (int r1 = 0; r1 < n1; r1++) {
                for (int c1 = 0; c1 < n2; c1++) {
                    for (int filter_idx = 0; filter_idx < n3; filter_idx++) {
                        Array1 delta_z = null;
                        Array1 delta_a;

                        Array1 a_sv = A.At3(r1, c1, filter_idx);
                        Array1 z_sv = null;
                        Array1 cost_deriv = layer.svdC_dA4.At3(r1, c1, filter_idx);

                        if (layer is ConvolutionalLayer) {
                            ConvolutionalLayer clayer = layer as ConvolutionalLayer;

                            z_sv = clayer.Z4.At3(r1, c1, filter_idx);
                            delta_z = 0.001 * z_sv;
                            clayer.Z4.Set3(r1, c1, filter_idx, clayer.Z4.At3(r1, c1, filter_idx) + delta_z);

                            A.Set3(r1, c1, filter_idx, clayer.Z4.At3(r1, c1, filter_idx).Map(Sys.Sigmoid));
                            delta_a = A.At3(r1, c1, filter_idx) - a_sv;
                        }
                        else {
                            PoolingLayer player = layer as PoolingLayer;

                            delta_a = 0.001 * a_sv;
                            A.Set3(r1, c1, filter_idx, A.At3(r1, c1, filter_idx) + delta_a);
                            player.Activation2 = (Array2)A.Reshape(MiniBatchSize, player.ImgRows * player.ImgCols * player.FilterCount);
                        }

                        for (Layer L = layer.NextLayer; L != null; L = L.NextLayer) {
                            L.forward2();
                        }

                        for (Layer L = LastLayer; L != null; L = L.PrevLayer) {
                            L.backward2(Y);
                            if (L == layer) {
                                break;
                            }
                        }

                        //-------------------- ΔC
                        Array1 deltaC = LastLayer.Cost - sv_cost;

                        //-------------------- ΔC ≒ Δa0 * δC/δa0
                        Array1 deltaC2 = delta_a * cost_deriv;

                        CheckEqual1(deltaC, deltaC2, out ret.dt[i_ret, 0, 0], out ret.dt[i_ret, 0, 1]);

                        if (layer is ConvolutionalLayer) {
                            ConvolutionalLayer clayer = layer as ConvolutionalLayer;

                            Array1 dA_dZ = z_sv.Map(Sys.SigmoidPrime);

                            //ΔC ≒ Δz0 * δC/δa0 * da0/dz0
                            Array1 deltaC3 = delta_z * cost_deriv * dA_dZ;

                            CheckEqual1(deltaC, deltaC3, out ret.dt[i_ret, 1, 0], out ret.dt[i_ret, 1, 1]);

                            clayer.Z4.Set3(r1, c1, filter_idx, z_sv);
                        }

                        A.Set3(r1, c1, filter_idx, a_sv);

                        i_ret++;
                    }
                }
            }

            Output((layer is ConvolutionalLayer ? "Cnv-dA" : "Pool-dA"), ret);

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.Activation2 = player.svActivation2.Clone();
            }
        }

        void Verify(Array2 X, Array2 Y) {
            double delta_param;

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = true;
            }

            Array1 sv_cost = LastLayer.Cost.Clone();

            foreach (Layer layer in Layers) {
                if (layer is FullyConnectedLayer) {
                    FullyConnectedLayer fl = layer as FullyConnectedLayer;

                    fl.svActivation2 = fl.Activation2.Clone();
                    fl.svZ2 = fl.Z2.Clone();
                    fl.svdC_dA2 = fl.dC_dA2.Clone();
                }
                else if (layer is Layer4) {
                    Layer4 l4 = layer as Layer4;
                    l4.svdC_dA4 = l4.dC_dA4.Clone();
                    l4.svActivation4 = l4.Activation4.Clone();
                    if (layer is PoolingLayer) {
                        PoolingLayer player = layer as PoolingLayer;
                        player.svActivation2 = player.Activation2.Clone();
                    }
                    else if(layer is ConvolutionalLayer) {
                        ConvolutionalLayer cnv_layer = layer as ConvolutionalLayer;

                        cnv_layer.svZ4 = cnv_layer.Z4.Clone();
                    }
                }
            }

            for (Layer layer = LastLayer; layer != null; layer = layer.PrevLayer) {
                Array3 ret = null;
                int i_ret;

                if (layer is FullyConnectedLayer) {
                    FullyConnectedLayer fl = layer as FullyConnectedLayer;

                    if (DoVerifyDeltaActivation2) {

                        VerifyDeltaActivation2(X, Y, sv_cost, fl);
                    }

                    if (DoVerifySub2) {

                        ret = new Array3(fl.Bias.Length + fl.Weight.dt.Length, 3, 2);

                        for (int i = 0; i < fl.Bias.Length; i++) {
                            double sv_param = fl.Bias[i];

                            delta_param = fl.Bias[i] * 0.001;
                            fl.Bias[i] += delta_param;

                            Array1 nabla = fl.NablaBiases.Col(i);
                            VerifySub2(X, Y, sv_cost, fl, delta_param, nabla, ret.dt, i);
                            fl.Bias[i] = sv_param;
                        }

                        for (int r = 0; r < fl.Weight.nRow; r++) {
                            for (int c = 0; c < fl.Weight.nCol; c++) {
                                double sv_param = fl.Weight[r, c];

                                delta_param = fl.Weight[r, c] * 0.001;
                                fl.Weight[r, c] += delta_param;

                                Array1 nabla = fl.NablaWeights.Depth(r, c);
                                VerifySub2(X, Y, sv_cost, fl, delta_param, nabla, ret.dt, fl.Bias.Length + r * fl.Weight.nCol + c);
                                fl.Weight[r, c] = sv_param;
                            }
                        }
                    }
                }
                else if (layer is Layer4) {
                    if (DoVerifyDeltaActivation4) {

                        VerifyDeltaActivation4(X, Y, sv_cost, layer as Layer4);
                    }

                    if (DoVerifySub4 && layer is ConvolutionalLayer) {
                        ConvolutionalLayer cnv_layer = layer as ConvolutionalLayer;

                        ret = new Array3(cnv_layer.Bias.Length, 3, 2);
                        i_ret = 0;

                        for (int filter_idx = 0; filter_idx < cnv_layer.Bias.Length; filter_idx++) {
                            Array1 nabla = cnv_layer.NablaBiases.Col(filter_idx);

                            double sv_param = cnv_layer.Bias[filter_idx];

                            delta_param = sv_param * 0.001;
                            delta_param = nabla[0] * 0.001;
                            cnv_layer.Bias[filter_idx] += delta_param;

                            VerifySub4(X, Y, sv_cost, cnv_layer, filter_idx, -1, -1, delta_param, nabla, ret.dt, i_ret);
                            cnv_layer.Bias[filter_idx] = sv_param;
                            i_ret++;
                        }
                        Output("Cnv-Bias", ret);

                        ret = new Array3(cnv_layer.Weight3.dt.Length, 3, 2);
                        i_ret = 0;

                        // すべてのフィルターに対し
                        for (int filter_idx = 0; filter_idx < cnv_layer.FilterCount; filter_idx++) {

                            // フィルターの行に対し
                            for (int r2 = 0; r2 < cnv_layer.FilterSize; r2++) {

                                // フィルターの列に対し
                                for (int c2 = 0; c2 < cnv_layer.FilterSize; c2++) {
                                    Array1 nabla = new Array1( from batch_idx in Enumerable.Range(0, MiniBatchSize) select cnv_layer.NablaWeight4[batch_idx, filter_idx, r2, c2] );

                                    double sv_param = cnv_layer.Weight3[filter_idx, r2, c2];

                                    delta_param = sv_param * 0.00001;
                                    delta_param = nabla[0] * 0.001;
                                    cnv_layer.Weight3[filter_idx, r2, c2] += delta_param;

                                    VerifySub4(X, Y, sv_cost, cnv_layer, filter_idx, r2, c2, delta_param, nabla, ret.dt, i_ret);

                                    cnv_layer.Weight3[filter_idx, r2, c2] = sv_param;
                                    i_ret++;
                                }
                            }
                        }
                        Output("Cnv-Weight", ret);
                    }
                }
            }

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = false;
            }
        }

        void VerifyResult(Array2 X, Array2 Y, double eta, int epoch_idx, int[] idxes, int mini_batch_cnt, int mini_batch_idx) {
            // パラメータの更新前の最後のレイヤーの出力を保存する。
            double[,] last_a = new double[MiniBatchSize, 10];
            for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                for (int c = 0; c < 10; c++) {
                    last_a[batch_idx, c] = LastLayer.Activation2[batch_idx, c];
                }
            }

            double eta2 = eta / MiniBatchSize;

            foreach (Layer layer in Layers) {
                layer.UpdateParameter(eta2);
            }

            foreach (Layer layer in Layers) {
                layer.forward2();
            }

            for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                var idx = idxes[mini_batch_idx * MiniBatchSize + batch_idx];
                int n = TrainLabel[idx];
                for (int c = 0; c < 10; c++) {
                    double diff = LastLayer.Activation2[batch_idx, c] - last_a[batch_idx, c];
                    if (c != n) {
                        diff = -diff;
                    }
                    DiffA[mini_batch_idx, batch_idx, c] = diff;
                }
            }

            if (mini_batch_idx != 0 && mini_batch_idx % 100 == 0) {
                StringWriter sw = new StringWriter();

                for (int mini_batch_idx_2 = 0; mini_batch_idx_2 <= mini_batch_idx; mini_batch_idx_2++) {
                    for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                        for (int c = 0; c < 10; c++) {
                            sw.Write(",{0}", DiffA[mini_batch_idx_2, batch_idx, c]);
                        }
                        sw.WriteLine("");
                    }
                    sw.WriteLine("");
                }
                try {
                    File.WriteAllText(string.Format("DiffA-{0}.csv", epoch_idx), sw.ToString());
                }
                catch (Exception) { }
            }
        }

        void VerifyLastActivation(Array2 X, Array2 Y, double eta, int epoch_idx, int[] idxes, int mini_batch_cnt, int mini_batch_idx) {

            for (int c = 0; c < 10; c++) {
                LastActivation[epoch_idx, mini_batch_idx, c] = LastLayer.Activation2[0, c];
            }

            if (mini_batch_idx % 100 == 0) {
                StringWriter sw = new StringWriter();

                for (int k = 0; k <= mini_batch_idx; k++) {
                    var idx = idxes[k * MiniBatchSize + 0];
                    int n = TrainLabel[idx];
                    for (int c = 0; c < 10; c++) {
                        if (c == n) {
                            sw.Write(",");
                        }
                        else {
                            sw.Write(",{0}", LastActivation[epoch_idx, k, c]);
                        }
                    }
                    sw.Write(",{0},{1}", LastActivation[epoch_idx, k, n], n);
                    sw.WriteLine();
                }

                try {
                    File.WriteAllText(string.Format("LastActivation-{0}.csv", epoch_idx), sw.ToString());
                }
                catch (Exception) { }
            }

            if (mini_batch_idx % 1000 == 0) {
                Debug.WriteLine("mini batch idx {0}", mini_batch_idx);
            }
        }

        void CheckEqual1(Array1 A, Array1 B, out double diff, out double ratio) {
            diff = (A - B).Map(Math.Abs).Max();
            ratio = A.Apply((double x, double y) => Math.Abs((y - x) * (x == 0 ? 1 : 1 / x)), B).Max();
        }

        void CheckEqual2(Array2 A, Array2 B, out double diff, out double ratio) {
            diff = (A - B).Map(Math.Abs).Max();
            ratio = A.Apply((double x, double y) => Math.Abs(y - x) * (x == 0 ? 1 : 1 / x), B).Max();
        }

    }
}