using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;

namespace MachineLearning {
    public partial class Network {
        public static int MiniBatchSize = 10;
        public static double Eta = 3.0;        // 10.0
        public static double Eta2 = Eta / MiniBatchSize;

        public static bool DoVerifyFull = true;
        public static bool DoVerifyConv = true;
        public static bool DoVerifySingleParam = false;
        public static bool DoVerifyMultiParam = false;
        public static bool DoVerifyMultiParamAll = false;
        public static bool DoVerifyDeltaActivation2 = false;
        public static bool DoVerifyDeltaActivation4 = false;
        public static bool DoVerifyUpdateParameter = true;

        Array3 LastActivation;
        Array3 DiffA;
        Array2 diffCosts;
        Array3 Result;

        void Output(string path, Array3 ret) {
            File.WriteAllText(path + ".csv", ret.ToString());
        }

        void VerifySingleParam2(Array2 X, Array2 Y, Array1 sv_cost, FullyConnectedLayer layer, double delta_param, Array1 nabla, double[,,] ret, int i_ret) {
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
            Array1 dP_dC_dP = delta_param * nabla;
            CheckEqual1(dP_dC_dP, dCost, out ret[i_ret, 0, 0], out ret[i_ret, 0, 1]);

            // ΔC ≒ ΔA * dC/dA
            CheckEqual1(dActivation2_dC_dA, dCost, out ret[i_ret, 1, 0], out ret[i_ret, 1, 1]);

            Array2 dA_dZ = layer.svZ2.Map(Sys.SigmoidPrime);
            Array2 dZ_dA_dZ = dZ * dA_dZ;

            // ΔA ≒ ΔZ * dA/dZ
            CheckEqual2(dZ_dA_dZ, dActivation2, out ret[i_ret, 2, 0], out ret[i_ret, 2, 1]);
        }

        void VerifySingleParam4(Array2 X, Array2 Y, Array1 sv_cost, ConvolutionalLayer layer, int batch_idx, double delta_param, Array1 dC_dP, double[,,] ret, int i_ret) {
            for (Layer l = layer; l != null; l = l.NextLayer) {
                l.forward2();
            }

            for (Layer l = LastLayer; ; l = l.PrevLayer) {
                l.backward2(Y);
                if (l == layer) {
                    break;
                }
            }

            // ΔZ
            Array4 dZ = layer.Z4 - layer.svZ4;

            // ΔA
            Array4 dA = layer.Activation4 - layer.svActivation4;

            int row_size = dA.dt.GetLength(0);
            int col_size = dA.dt.GetLength(1) * dA.dt.GetLength(2) * dA.dt.GetLength(3);

            // ΔA * dC/dA
            Array2 dA_dC_dA = (dA * layer.dC_dA4).Reshape(row_size, col_size) as Array2;

            // Σ ΔAi * dC/dAi
            Array1 sum_dA_dC_dA = dA_dC_dA.SumRow();

            // ΔC
            Array1 dC = LastLayer.Cost - sv_cost;

            // ΔP * δC/δP
            //Array1 dP_dC_dP = delta_param * dC_dP;
            double dP_dC_dP = delta_param * dC_dP[batch_idx];

            // ΔC ≒ ΔP * δC/δP 
            CheckEqual0(dC[batch_idx], dP_dC_dP, out ret[i_ret, 0, 0], out ret[i_ret, 0, 1]);

            // ΔC ≒ ΔA * dC/dA
            CheckEqual1(dC, sum_dA_dC_dA, out ret[i_ret, 1, 0], out ret[i_ret, 1, 1]);

            // dA/dZ
            Array4 dA_dZ = layer.svZ4.Map(Sys.SigmoidPrime);

            // ΔZ * dA/dZ
            Array4 dZ_dA_dZ = dZ * dA_dZ;

            // ΔA ≒ ΔZ * dA/dZ
            CheckEqual2(dA.Reshape(row_size, col_size) as Array2, dZ_dA_dZ.Reshape(row_size, col_size) as Array2, out ret[i_ret, 2, 0], out ret[i_ret, 2, 1]);
        }


        void VerifyMultiParam(Array2 X, Array2 Y, Array1 sv_cost, Layer layer, int batch_idx, Array1 dP, Array1 dC_dP, double[,,] ret, int i_ret) {
            for (Layer l = layer; l != null; l = l.NextLayer) {
                l.forward2();
            }

            for (Layer l = LastLayer; ; l = l.PrevLayer) {
                l.backward2(Y);
                if (l == layer) {
                    break;
                }
            }

            Array2 Z, svZ, A, svA, svdC_dA;

            if(layer is FullyConnectedLayer) {

                FullyConnectedLayer full_layer = layer as FullyConnectedLayer;

                Z = full_layer.Z2;
                svZ = full_layer.svZ2;
                A = full_layer.Activation2;
                svA = full_layer.svActivation2;
                svdC_dA = full_layer.svdC_dA2;
            }
            else {
                ConvolutionalLayer cnv_layer = layer as ConvolutionalLayer;

                int row_size = cnv_layer.Z4.dt.GetLength(0);
                int col_size = cnv_layer.Z4.dt.GetLength(1) * cnv_layer.Z4.dt.GetLength(2) * cnv_layer.Z4.dt.GetLength(3);

                Z = cnv_layer.Z4.Reshape(row_size, col_size) as Array2;
                svZ = cnv_layer.svZ4.Reshape(row_size, col_size) as Array2;
                A = cnv_layer.Activation4.Reshape(row_size, col_size) as Array2;
                svA= cnv_layer.svActivation4.Reshape(row_size, col_size) as Array2;
                svdC_dA= cnv_layer.svdC_dA4.Reshape(row_size, col_size) as Array2;
            }

            // ΔZ
            Array2 dZ = Z - svZ;

            // ΔA
            Array2 dA = A - svA;

            // ΔA * dC/dA
            Array2 dA_dC_dA = dA * svdC_dA;

            // Σ ΔAi * dC/dAi
            Array1 sum_dA_dC_dA = dA_dC_dA.SumRow();

            // ΔC
            Array1 dC = LastLayer.Cost - sv_cost;

            // Σ ΔPi * δC/δPi
            double dP_dC_dP = (dP * dC_dP).Sum();

            // ΔC ≒ ΔP * δC/δP 
            CheckEqual0(dC.dt[batch_idx], dP_dC_dP, out ret[i_ret, 0, 0], out ret[i_ret, 0, 1]);

            // ΔC ≒ Σ ΔAi * dC/dAi
            CheckEqual1(dC, sum_dA_dC_dA, out ret[i_ret, 1, 0], out ret[i_ret, 1, 1]);

            // dA/dZ
            Array2 dA_dZ = svZ.Map(Sys.SigmoidPrime);

            // ΔZ * dA/dZ
            Array2 dZ_dA_dZ = dZ * dA_dZ;

            // ΔA ≒ ΔZ * dA/dZ
            CheckEqual2(dA, dZ_dA_dZ, out ret[i_ret, 2, 0], out ret[i_ret, 2, 1]);
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

        void SaveParamData(out Array1 sv_cost) {
            sv_cost = LastLayer.Cost.Clone();

            foreach (Layer layer in Layers) {
                layer.SaveParam();
                if (layer is FullyConnectedLayer) {
                    FullyConnectedLayer fl = layer as FullyConnectedLayer;

                    fl.svActivation2 = fl.Activation2.Clone();
                    fl.svZ2 = fl.Z2.Clone();
                }
                else if (layer is Layer4) {
                    Layer4 l4 = layer as Layer4;
                    l4.svdC_dA4 = l4.dC_dA4.Clone();
                    l4.svActivation4 = l4.Activation4.Clone();
                    if (layer is PoolingLayer) {
                        PoolingLayer player = layer as PoolingLayer;
                        player.svActivation2 = player.Activation2.Clone();
                    }
                    else if (layer is ConvolutionalLayer) {
                        ConvolutionalLayer cnv_layer = layer as ConvolutionalLayer;

                        cnv_layer.svZ4 = cnv_layer.Z4.Clone();
                    }
                }
            }
        }

        void VerifyUpdateParameter(Array2 X, Array2 Y, int epoch_idx, int mini_batch_cnt, int mini_batch_idx) {
            if (mini_batch_idx % 100 != 0) {

                foreach (Layer layer in Layers) {
                    int param_len = layer.ParamLength();
                    if (param_len != 0) {

                        for (int param_i = 0; param_i < param_len; param_i++) {

                            double dC_dP = layer.dC_dPByParamIdxAvg(param_i);
                            double delta_param = dC_dP * -0.1;
                            layer.ParamSet(param_i, layer.ParamAt(param_i) + delta_param);
                        }
                    }
                }
                return;
            }


            if (diffCosts == null || diffCosts.nRow != mini_batch_cnt) {
                diffCosts = new Array2(mini_batch_cnt, 4);
                Result = new Array3(mini_batch_cnt, Y.nRow, 10);
            }

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = true;
            }

            Array1 sv_cost;
            SaveParamData(out sv_cost);

            int param_len_sum = (from layer in Layers select layer.ParamLength()).Sum();
            Array1 dPall = new Array1(param_len_sum);
            Array1 dC_dPall = new Array1(param_len_sum);
            int param_idx_all = 0;
            foreach (Layer layer in Layers) {

                int param_len = layer.ParamLength();
                if (param_len != 0) {

                    layer.RestoreParam();

                    Array1 dP = new Array1(param_len);

                    for (int param_i = 0; param_i < param_len; param_i++) {

                        double dC_dP = layer.dC_dPByParamIdxAvg(param_i);
                        dP[param_i] = dC_dP * -0.1;
                        layer.ParamSet(param_i, layer.ParamAt(param_i) + dP[param_i]);

                        dPall[param_idx_all] = dP[param_i];
                        dC_dPall[param_idx_all] = dC_dP;
                        param_idx_all++;
                    }
                }
            }

            foreach (Layer l in Layers) {
                l.forward2();
            }

            for (Layer l = LastLayer; l != null; l = l.PrevLayer) {
                l.backward2(Y);
            }

            // ΔC
            Array1 dC = LastLayer.Cost - sv_cost;

            // Σ ΔPi * δC/δPi
            double dP_dC_dP = (dPall * dC_dPall).Sum();

            // ΔC ≒ ΔP * δC/δP 
            double c_avg = LastLayer.Cost.Avg();
            double dc_avg = dC.Avg();
            double diff, ratio;
            CheckEqual0(dc_avg, dP_dC_dP, out diff, out ratio);

            diffCosts.dt[mini_batch_idx, 0] = c_avg;
            diffCosts.dt[mini_batch_idx, 1] = dc_avg;
            diffCosts.dt[mini_batch_idx, 2] = diff;
            diffCosts.dt[mini_batch_idx, 3] = ratio;

            for(int r = 0; r < Y.nRow; r++) {
                for (int c = 0; c < 10; c++) {
                    Result.dt[mini_batch_idx, r, c] = LastLayer.svdC_dA2.dt[r, c];
                }
            }

            if (mini_batch_idx % 10 == 0) {
                Debug.WriteLine("{0} cost:{1}  dC avg:{2} diff:{3} ratio:{4}", mini_batch_idx, c_avg, dc_avg, diff, ratio);

                if (mini_batch_idx % 100 == 0) {

                    StringWriter sw = new StringWriter();

                    for (int r = 0; r <= mini_batch_idx; r++) {
                        for (int c = 0; c < 4; c++) {
                            sw.Write(",{0}", diffCosts.dt[r, c]);
                        }
                        sw.WriteLine();
                    }

                    try {
                        File.WriteAllText("Cost.csv", sw.ToString());

                    }
                    catch (Exception) { }

                    sw = new StringWriter();

                    for (int d = 0; d <= mini_batch_idx; d++) {
                        for(int r = 0; r < Y.nRow; r++) {
                            for (int c = 0; c < 10; c++) {
                                sw.Write(",{0}", Result.dt[d, r, c]);
                            }
                            sw.WriteLine();
                        }
                    }

                    try {
                        File.WriteAllText("Result.csv", sw.ToString());
                    }
                    catch (Exception) { }
                }
            }

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = false;
            }

            if (mini_batch_idx % 500 == 0) {
                int e = Evaluate();
                Debug.WriteLine("Epoch {0}: {1} / {2}", epoch_idx, e, TestImage.GetLength(0));
            }
        }

        void Verify(Array2 X, Array2 Y) {
            Array3 ret = null;
            int i_ret;
            double delta_param;

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = true;
            }

            Array1 sv_cost;
            SaveParamData(out sv_cost);

            if (DoVerifyMultiParamAll) {
                ret = new Array3(MiniBatchSize, 3, 2);
                i_ret = 0;

                for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {

                    int param_len_sum = (from layer in Layers select layer.ParamLength()).Sum();
                    Array1 dPall = new Array1(param_len_sum);
                    Array1 dC_dPall = new Array1(param_len_sum);
                    int param_idx_all = 0;
                    foreach (Layer layer in Layers) {

                        int param_len = layer.ParamLength();
                        if (param_len != 0) {

                            layer.RestoreParam();

                            Array1 dP = new Array1(param_len);
                            Array1 dC_dP = layer.dC_dPs(batch_idx);
                            Debug.Assert(dC_dP.Length == param_len);

                            for (int param_i = 0; param_i < param_len; param_i++) {

                                dP[param_i] = dC_dP[param_i] * 0.001;
                                layer.ParamSet(param_i, layer.ParamAt(param_i) + dP[param_i]);

                                dPall[param_idx_all] = dP[param_i];
                                dC_dPall[param_idx_all] = dC_dP[param_i];
                                param_idx_all++;
                            }
                        }
                    }

                    foreach (Layer l in Layers) {
                        l.forward2();
                    }

                    for (Layer l = LastLayer; l != null; l = l.PrevLayer) {
                        l.backward2(Y);
                    }

                    // ΔC
                    Array1 dC = LastLayer.Cost - sv_cost;

                    // Σ ΔPi * δC/δPi
                    double dP_dC_dP = (dPall * dC_dPall).Sum();

                    // ΔC ≒ ΔP * δC/δP 
                    CheckEqual0(dC.dt[batch_idx], dP_dC_dP, out ret.dt[i_ret, 0, 0], out ret.dt[i_ret, 0, 1]);
                    i_ret++;
                }
                Output("Multi-Param-All", ret);
            }

            foreach (Layer layer in Layers) {
                layer.RestoreParam();
            }

            for (Layer layer = LastLayer; layer != null; layer = layer.PrevLayer) {
                int layer_idx = Layers.ToList().IndexOf(layer);

                if (DoVerifySingleParam && (DoVerifyFull && layer is FullyConnectedLayer || DoVerifyConv && layer is ConvolutionalLayer)) {

                    //-------------------------------------------------- 単一パラメータ

                    for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                        int param_len = layer.ParamLength();
                        ret = new Array3(param_len, 3, 2);
                        i_ret = 0;

                        for (int param_i = 0; param_i < param_len; ) {
                            layer.RestoreParam();

                            Array1 dC_dP = layer.dC_dPByParamIdx(param_i);

                            double sv_param = layer.ParamAt(param_i);

                            delta_param = dC_dP[batch_idx] * 0.001;
                            layer.ParamSet(param_i, sv_param + delta_param);

                            if (layer is FullyConnectedLayer) {

                                VerifySingleParam2(X, Y, sv_cost, layer as FullyConnectedLayer, delta_param, dC_dP, ret.dt, i_ret);
                            }
                            else {

                                VerifySingleParam4(X, Y, sv_cost, layer as ConvolutionalLayer, batch_idx, delta_param, dC_dP, ret.dt, i_ret);
                            }

                            i_ret++;

                            layer.ParamSet(param_i, sv_param);

                            if(param_len < 1000) {

                                param_i++;
                            }
                            else {

                                param_i += param_len / 1000;
                            }
                        }

                        Output(layer.GetType().Name.Substring(0, 4) + "-Single-Param", ret);
                        if (batch_idx == 0) {

                            break;
                        }
                    }
                    layer.RestoreParam();

                }

                //-------------------------------------------------- 全パラメータ
                if (DoVerifyMultiParam && (DoVerifyFull && layer is FullyConnectedLayer || DoVerifyConv && layer is ConvolutionalLayer)) {

                    int param_len = layer.ParamLength();
                    ret = new Array3(MiniBatchSize, 3, 2);
                    i_ret = 0;
                    for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {

                        layer.RestoreParam();

                        Array1 dP = new Array1(param_len);
                        Array1 dC_dP = layer.dC_dPs(batch_idx);
                        Debug.Assert(dC_dP.Length == param_len);

                        for (int param_i = 0; param_i < param_len; param_i++) {

                            dP[param_i] = dC_dP[param_i] * 0.001;
                            layer.ParamSet(param_i, layer.ParamAt(param_i) + dP[param_i]);
                        }

                        VerifyMultiParam(X, Y, sv_cost, layer, batch_idx, dP, dC_dP, ret.dt, i_ret);
                        i_ret++;
                    }
                    Output(string.Format("{0} {1} Multi-Param", layer.GetType().Name.Substring(0, 4), layer_idx), ret);
                    layer.RestoreParam();
                }

                if (layer is FullyConnectedLayer) {
                    FullyConnectedLayer fl = layer as FullyConnectedLayer;

                    if (DoVerifyDeltaActivation2) {

                        VerifyDeltaActivation2(X, Y, sv_cost, fl);
                    }
                }
                else if (layer is Layer4) {
                    if (DoVerifyDeltaActivation4) {

                        VerifyDeltaActivation4(X, Y, sv_cost, layer as Layer4);
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
                layer.UpdateParameter();
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

        void CheckEqual0(double A, double B, out double diff, out double ratio) {
            diff = Math.Abs(A - B);
            ratio = Math.Abs( (B - A) * (A == 0 ? 1 : 1 / A) );
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