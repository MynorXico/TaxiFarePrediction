using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// To create a learning pipeline
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;


namespace TaxiFarePrediction
{

    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = Train();
        }

        public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
        {
            LearningPipeline pipeline = new LearningPipeline();

        }
    }
}
