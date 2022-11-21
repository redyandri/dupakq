import React, { useState } from "react";
import { render } from "react-dom";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import networkgraph from "highcharts/modules/networkgraph";

networkgraph(Highcharts);

export default function NetGraph(){
  var [pointClick, handlePointClick] = useState(null);
  const [chartOptions, setChartOptions] = useState({
    chart: {
      type: "networkgraph"
    },
    plotOptions: {
      networkgraph: {
        layoutAlgorithm: {
          enableSimulation: false
        },
        point: {
          events: {
            click(e) {
              handlePointClick(e.point);
            }
          }
        }
      }
    },
    series: [
      {
        dataLabels: {
          enabled: true
        },
        marker: {
          radius: 35
        },
        data: [
          {
            from: "n1",
            to: "n2"
          },
          {
            from: "n2",
            to: "n3"
          },
          {
            from: "n3",
            to: "n4"
          },
          {
            from: "n4",
            to: "n5"
          },
          {
            from: "n5",
            to: "n1"
          }
        ]
      }
    ]
  });

  handlePointClick = e => {
    console.log(e);
  };


  return (
    <div>
      <HighchartsReact highcharts={Highcharts} options={chartOptions} />
    </div>
  );
};

// export default NetGraph;


