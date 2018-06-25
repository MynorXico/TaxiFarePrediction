using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    static class TestTrip
    {
        internal static readonly TaxiTrip Trip1 = new TaxiTrip
        {
            VendorId = "VTS",
            RateCode = "1",
            PassengerCount = 1,
            TripDistance = 10.33f,
            PaymentType = "CSH",
            FareAmount = 0 // Predict it. actual = 29.5
        };
    }
}
