function [o, A, M, a, alpha, b, lu] = DataLoading(problem, D)

    switch problem

        case 1

            % lu: define the upper and lower bounds of the variables
            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            % Load the data for this test function
            load sphere_func_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 2

            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            load schwefel_102_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 3

            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            load high_cond_elliptic_rot_data
            A = []; a = []; alpha = []; b = [];

            if D == 2, load elliptic_M_D2,
            elseif D == 10, load elliptic_M_D10,
            elseif D == 30, load elliptic_M_D30,
            elseif D == 50, load elliptic_M_D50,
            end

        case 4

            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            load schwefel_102_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 5

            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            load schwefel_206_data
            M = []; a = []; alpha = []; b = [];

        case 6

            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            load rosenbrock_func_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 7

            lu = [0 * ones(D, 1), 600 * ones(D, 1)];
            load griewank_func_data
            A = []; a = []; alpha = []; b = [];

            c = 3;
            if D == 2, load griewank_M_D2,
            elseif D == 10, load griewank_M_D10,
            elseif D == 30, load griewank_M_D30,
            elseif D == 50, load griewank_M_D50,
            end

        case 8

            lu = [-32 * ones(D, 1), 32 * ones(D, 1)];
            load ackley_func_data
            A = []; a = []; alpha = []; b = [];

            if D == 2, load ackley_M_D2,
            elseif D == 10, load ackley_M_D10,
            elseif D == 30, load ackley_M_D30,
            elseif D == 50, load ackley_M_D50,
            end

        case 9

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load rastrigin_func_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 10

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load rastrigin_func_data
            A = []; a = []; alpha = []; b = [];
            if D == 2, load rastrigin_M_D2,
            elseif D == 10, load rastrigin_M_D10,
            elseif D == 30, load rastrigin_M_D30,
            elseif D == 50, load rastrigin_M_D50,
            end

        case 11

            lu = [-0.5 * ones(D, 1), 0.5 * ones(D, 1)];
            load weierstrass_data
            A = []; a = []; alpha = []; b = [];
            if D == 2, load weierstrass_M_D2,
            elseif D == 10, load weierstrass_M_D10,
            elseif D == 30, load weierstrass_M_D30,
            elseif D == 50, load weierstrass_M_D50,
            end

        case 12

            lu = [-pi * ones(D, 1), pi * ones(D, 1)];
            load schwefel_213_data
            A = []; M = []; o = [];

        case 13

            lu = [-3 * ones(D, 1), 1 * ones(D, 1)];
            load EF8F2_func_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 14

            lu = [-100 * ones(D, 1), 100 * ones(D, 1)];
            load E_ScafferF6_func_data
            if D == 2, load E_ScafferF6_M_D2,
            elseif D == 10, load E_ScafferF6_M_D10,
            elseif D == 30, load E_ScafferF6_M_D30,
            elseif D == 50, load E_ScafferF6_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 15

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func1_data
            A = []; M = []; a = []; alpha = []; b = [];

        case 16

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func1_data
            if D == 2, load hybrid_func1_M_D2,
            elseif D == 10, load hybrid_func1_M_D10,
            elseif D == 30, load hybrid_func1_M_D30,
            elseif D == 50, load hybrid_func1_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 17

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func1_data
            if D == 2, load hybrid_func1_M_D2,
            elseif D == 10, load hybrid_func1_M_D10,
            elseif D == 30, load hybrid_func1_M_D30,
            elseif D == 50, load hybrid_func1_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 18

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func2_data
            if D == 2, load hybrid_func2_M_D2,
            elseif D == 10, load hybrid_func2_M_D10,
            elseif D == 30, load hybrid_func2_M_D30,
            elseif D == 50, load hybrid_func2_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 19

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func2_data
            if D == 2, load hybrid_func2_M_D2,
            elseif D == 10, load hybrid_func2_M_D10,
            elseif D == 30, load hybrid_func2_M_D30,
            elseif D == 50, load hybrid_func2_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 20

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func2_data
            if D == 2, load hybrid_func2_M_D2,
            elseif D == 10, load hybrid_func2_M_D10,
            elseif D == 30, load hybrid_func2_M_D30,
            elseif D == 50, load hybrid_func2_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 21

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func3_data
            if D == 2, load hybrid_func3_M_D2,
            elseif D == 10, load hybrid_func3_M_D10,
            elseif D == 30, load hybrid_func3_M_D30,
            elseif D == 50, load hybrid_func3_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 22

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func3_data
            if D == 2, load hybrid_func3_HM_D2,
            elseif D == 10, load hybrid_func3_HM_D10,
            elseif D == 30, load hybrid_func3_HM_D30,
            elseif D == 50, load hybrid_func3_HM_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 23

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func3_data
            if D == 2, load hybrid_func3_M_D2,
            elseif D == 10, load hybrid_func3_M_D10,
            elseif D == 30, load hybrid_func3_M_D30,
            elseif D == 50, load hybrid_func3_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 24

            lu = [-5 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func4_data
            if D == 2, load hybrid_func4_M_D2,
            elseif D == 10, load hybrid_func4_M_D10,
            elseif D == 30, load hybrid_func4_M_D30,
            elseif D == 50, load hybrid_func4_M_D50,
            end
            A = []; a = []; alpha = []; b = [];

        case 25

            lu = [2 * ones(D, 1), 5 * ones(D, 1)];
            load hybrid_func4_data
            if D == 2, load hybrid_func4_M_D2,
            elseif D == 10, load hybrid_func4_M_D10,
            elseif D == 30, load hybrid_func4_M_D30,
            elseif D == 50, load hybrid_func4_M_D50,
            end
            A = []; a = []; alpha = []; b = [];
    end
end

