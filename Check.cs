using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace MachineLearning {
    public partial class Network {

        void VerifySub(Array2 X, Array2 Y, double eta, Array1 sv_cost, FullyConnectedLayer layer, double delta_param, Array1 nabla, double[,,] ret, int i_ret) {
            for (Layer l = layer; l != null; l = l.NextLayer) {
                l.forward2();
            }

            for (Layer l = LastLayer; ; l = l.PrevLayer) {
                l.backward2(Y, eta);
                if (l == layer) {
                    break;
                }
            }

            Array2 dActivation2 = layer.Activation2 - layer.svActivation2;
            Array2 dZ = layer.Z - layer.svZ;
            Array1 dActivation2_CostDerivative = (dActivation2 * layer.CostDerivative2).SumRow();
            Array1 dCost = LastLayer.Cost - sv_cost;

            // Δparam nabla  ≒ ΔC
            Array1 delta_param_nabla = delta_param * nabla;
            CheckEqual1(delta_param_nabla, dCost, out ret[i_ret, 0, 0], out ret[i_ret, 0, 1]);

            // ΔA dC/dA ≒ ΔC
            CheckEqual1(dActivation2_CostDerivative, dCost, out ret[i_ret, 1, 0], out ret[i_ret, 1, 1]);

            Array2 sigmoid_prime = layer.svZ.Map(Sys.SigmoidPrime);
            Array2 dZ_sigmoid_prime = dZ * sigmoid_prime;

            // ΔZ dA/dZ ≒ ΔA
            CheckEqual2(dZ_sigmoid_prime, dActivation2, out ret[i_ret, 2, 0], out ret[i_ret, 2, 1]);
        }


        void VerifyDeltaActivation2(Array2 X, Array2 Y, double eta, Array1 sv_cost, FullyConnectedLayer layer) {
            Array2 A = layer.GetActivation2();

            Array2 Err2 = new Array2(A.nRow, A.nCol);
            Array2 Err3 = new Array2(A.nRow, A.nCol);

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = true;
            }

            for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                for (int k = 0; k < A.nCol; k++) {
                    double delta_z = 0;
                    double delta_a;

                    double a_sv = A[batch_idx, k];
                    double z_sv = 0;
                    double cost_deriv = layer.svCostDerivative2[batch_idx, k];

                    z_sv = layer.Z.dt[batch_idx, k];
                    delta_z = z_sv * 0.001;
                    layer.Z.dt[batch_idx, k] += delta_z;

                    A[batch_idx, k] = Sys.Sigmoid(layer.Z.dt[batch_idx, k]);
                    delta_a = A[batch_idx, k] - a_sv;

                    for (Layer L = layer.NextLayer; L != null; L = L.NextLayer) {
                        L.forward2();
                    }

                    for (Layer L = LastLayer; L != null; L = L.PrevLayer) {
                        L.backward2(Y, eta);
                        if (L == layer) {
                            break;
                        }
                    }

                    //-------------------- ΔC
                    double deltaC = LastLayer.Cost[batch_idx] - sv_cost[batch_idx];

                    //-------------------- ΔC ≒ Δa0 * δC/δa0
                    double deltaC2 = delta_a * cost_deriv;

                    Err2[batch_idx, k] = Math.Abs(deltaC2 - deltaC);

                    double sigmoid_prime_z = Sys.SigmoidPrime(z_sv);

                    //ΔC ≒ Δz0 * δC/δa0 * da0/dz0
                    double deltaC3 = delta_z * cost_deriv * sigmoid_prime_z;

                    Err3[batch_idx, k] = Math.Abs(deltaC3 - deltaC);

                    layer.Z.dt[batch_idx, k] = z_sv;

                    A[batch_idx, k] = a_sv;
                }
            }

            double max_err2 = Err2.Max();
            double max_err3 = Err3.Max();

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = false;
                player.Activation2 = player.svActivation2.Clone();
            }
        }

        void VerifyDeltaActivation4(Array2 X, Array2 Y, double eta, Array1 sv_cost, Layer4 layer) {
            Array4 A = layer.GetActivation4();
            int n0 = A.dt.GetLength(0);
            int n1 = A.dt.GetLength(1);
            int n2 = A.dt.GetLength(2);
            int n3 = A.dt.GetLength(3);

            Array4 Err2 = new Array4(A.Shape());
            Array4 Err3 = new Array4(A.Shape());

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = true;
            }

            for (int batch_idx = 0; batch_idx < MiniBatchSize; batch_idx++) {
                for (int r1 = 0; r1 < n1; r1++) {
                    for (int c1 = 0; c1 < n2; c1++) {
                        for (int filter_idx = 0; filter_idx < n3; filter_idx++) {
                            double delta_z = 0;
                            double delta_a;

                            double a_sv = A[batch_idx, r1, c1, filter_idx];
                            double z_sv = 0;
                            double cost_deriv = layer.svCostDerivative4[batch_idx, r1, c1, filter_idx];

                            if (layer is ConvolutionalLayer) {
                                ConvolutionalLayer clayer = layer as ConvolutionalLayer;

                                z_sv = clayer.Z.dt[batch_idx, r1, c1, filter_idx];
                                delta_z = z_sv * 0.001;
                                clayer.Z.dt[batch_idx, r1, c1, filter_idx] += delta_z;

                                A[batch_idx, r1, c1, filter_idx] = Sys.Sigmoid(clayer.Z.dt[batch_idx, r1, c1, filter_idx]);
                                delta_a = A[batch_idx, r1, c1, filter_idx] - a_sv;
                            }
                            else {
                                PoolingLayer player = layer as PoolingLayer;

                                delta_a = a_sv * 0.001;
                                A[batch_idx, r1, c1, filter_idx] += delta_a;
                                player.Activation2 = (Array2)A.Reshape(MiniBatchSize, player.ImgRows * player.ImgCols * player.FilterCount);
                            }

                            for (Layer L = layer.NextLayer; L != null; L = L.NextLayer) {
                                L.forward2();
                            }

                            for (Layer L = LastLayer; L != null; L = L.PrevLayer) {
                                L.backward2(Y, eta);
                                if (L == layer) {
                                    break;
                                }
                            }

                            //-------------------- ΔC
                            double deltaC = LastLayer.Cost[batch_idx] - sv_cost[batch_idx];

                            //-------------------- ΔC ≒ Δa0 * δC/δa0
                            double deltaC2 = delta_a * cost_deriv;

                            Err2[batch_idx, r1, c1, filter_idx] = Math.Abs(deltaC2 - deltaC);

                            if (layer is ConvolutionalLayer) {
                                ConvolutionalLayer clayer = layer as ConvolutionalLayer;

                                double sigmoid_prime_z = Sys.SigmoidPrime(z_sv);

                                //ΔC ≒ Δz0 * δC/δa0 * da0/dz0
                                double deltaC3 = delta_z * cost_deriv * sigmoid_prime_z;

                                Err3[batch_idx, r1, c1, filter_idx] = Math.Abs(deltaC3 - deltaC);

                                clayer.Z.dt[batch_idx, r1, c1, filter_idx] = z_sv;
                            }

                            A[batch_idx, r1, c1, filter_idx] = a_sv;
                        }
                    }
                }
            }

            double max_err2 = Err2.Max();
            double max_err3 = Err3.Max();

            foreach (PoolingLayer player in from x in Layers where x is PoolingLayer select x) {
                player.RetainMaxIdx = false;
                player.Activation2 = player.svActivation2.Clone();
            }
        }

        void Verify(Array2 X, Array2 Y, double eta) {
            double delta_param;

            Array1 sv_cost = LastLayer.Cost.Clone();

            foreach (Layer layer in Layers) {
                if (layer is FullyConnectedLayer) {
                    FullyConnectedLayer fl = layer as FullyConnectedLayer;

                    fl.svActivation2 = fl.Activation2.Clone();
                    fl.svZ = fl.Z.Clone();
                    fl.svCostDerivative2 = fl.CostDerivative2.Clone();
                }
                else if (layer is Layer4) {
                    Layer4 l4 = layer as Layer4;
                    l4.svCostDerivative4 = l4.CostDerivative4.Clone();
                    if (layer is PoolingLayer) {
                        PoolingLayer player = layer as PoolingLayer;
                        player.svActivation2 = player.Activation2.Clone();
                    }
                }
            }

            for (Layer layer = LastLayer; layer != null; layer = layer.PrevLayer) {
                if (layer is FullyConnectedLayer) {
                    FullyConnectedLayer fl = layer as FullyConnectedLayer;

                    VerifyDeltaActivation2(X, Y, eta, sv_cost, fl);

                    Array3 ret = new Array3(fl.Bias.Length + fl.Weight.dt.Length, 3, 2);

                    for (int i = 0; i < fl.Bias.Length; i++) {
                        double sv_param = fl.Bias[i];

                        delta_param = fl.Bias[i] * 0.001;
                        fl.Bias[i] += delta_param;

                        Array1 nabla = fl.NablaBiases.Col(i);
                        VerifySub(X, Y, eta, sv_cost, fl, delta_param, nabla, ret.dt, i);
                        fl.Bias[i] = sv_param;
                    }

                    for (int r = 0; r < fl.Weight.nRow; r++) {
                        for (int c = 0; c < fl.Weight.nCol; c++) {
                            double sv_param = fl.Weight[r, c];

                            delta_param = fl.Weight[r, c] * 0.001;
                            fl.Weight[r, c] += delta_param;

                            Array1 nabla = fl.NablaWeights.Depth(r, c);
                            VerifySub(X, Y, eta, sv_cost, fl, delta_param, nabla, ret.dt, fl.Bias.Length + r * fl.Weight.nCol + c);
                            fl.Weight[r, c] = sv_param;
                        }
                    }

                    double err = ret.Map(Math.Abs).Max();
                    double avg = ret.Map(Math.Abs).Sum() / ret.dt.Length;
                    Debug.WriteLine("検証 --------------------------------------------------------------------------------");
                    Debug.WriteLine(ret.ToString());
                }
                else if (layer is Layer4) {
                    VerifyDeltaActivation4(X, Y, eta, sv_cost, layer as Layer4);
                }
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