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

        static async Task Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            #region Optional
            /*
                LearningPipeline pipeline = new LearningPipeline();
                pipeline.Add(new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','));
                pipeline.Add(new ColumnCopier(("FareAmount", "Label")));
                pipeline.Add(new CategoricalOneHotVectorizer("VendorId",
                                                            "RateCode",
                                                            "PaymentType"));

                // Indicate which columns are features
                pipeline.Add(new ColumnConcatenator("Features",
                    "VendorId",
                    "RateCode",
                    "Passengercount",
                    "TripDistance",
                    "PaymentType"));

                pipeline.Add(new FastTreeRegressor());
            */
            #endregion

            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer(
                    "VendorId",
                    "RateCode",
                    "PaymentType"),
                new ColumnConcatenator(
                    "Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"),
                new FastTreeRegressor()
            };
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelpath);
            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"RMS = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");


            // Using the model for predictions
            TaxiTripFarePrediction prediction = model.Predict(TestTrip.Trip1);
            Console.WriteLine($"Predicted fare: {prediction.FareAmount}, actual fare: 29.5");
        }
    }
}
