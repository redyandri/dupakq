import Graph from 'react-vis-network-graph';
import React, { useState, memo, useEffect } from "react";
import GraphTable from './GraphTable'
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';

require('dotenv').config()

const API_URI = 'http://' + process.env.REACT_APP_API_HOST + ':' + process.env.REACT_APP_API_PORT


const stretch_style= { height: "100%" };
const item_style= { display: "flex", flexDirection: "column" }; // KEY CHANGES

function getWindowHeight() {
  const {innerHeight: height } = window;
  return height;
}

function RGV({ data, q }) {
  const tmp = {
    "distance": 0,
    "model": " ",
    "index": '-',
    "code": " ",
    "activity": " ",
    "level": " ",
    "credit": " "
  }

  const [graphHeight,setGraphHeight]=useState(0)
  const graphHeightPercentage=55;

  useEffect(() => {
    const gh=getWindowHeight()*(graphHeightPercentage/100);
    setGraphHeight(gh);
  },[]);

  const [graphdata, setGraphdata] = useState([tmp])

  const showGrapProp = async (idx) => {
    try {
      console.log('idx to query:', idx)
      const respnd = await fetch(API_URI + '/get_dupak_by_indexes/', {
        mode: "cors",
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          "Content-type": "application/json",
          "Access-Control-Allow-Origin": "*"
        },
        body: JSON.stringify({
          "q": idx
        })
      });
      const json_data = await respnd.json();
      const results = json_data.results;
      // console.log('results:',results)
      // console.log('results.edges:',results.edges)
      setGraphdata(results);



    } catch (error) {
      console.log('error:' + error);
    };
  };

  const events = {
    select: ({ nodes, edges }) => {
      // console.log("Selected nodes:" + nodes);
      // console.log("Selected edges:" + edges);
      let ids = []
      for (let i = 0; i < edges.length; i++) {
        try {
          let id = edges[i].split('_')[2];
          ids.push(id);
        } catch (e) {
          continue;
        }

        // console.log('id:',id)
      }
      if (ids.length > 0) {
        let ids2 = ids.join(';');
        showGrapProp(ids2);
      }

      // console.log('ids:',ids)
      // console.log('ids2:',ids2)

    },
    // selectEdge: (e) => {
    //   var { nodes, edges } = e;
    //   if (edges.length >= 1) {
    //     // console.log('selectEdge:' + edges + '\nNodes:' + nodes);
    //     let ids = []
    //     for (let i = 0; i < edges.length; i++) {
    //       let id = edges[i].split('_')[2]
    //       ids.push(id)
    //     }
    //     let ids2 = ids.join(';')
    //     showGrapProp(ids2);
    //   }
    // }
  };

  const options = {
    layout: {
      hierarchical: false
    },
    edges: {
      color: "#000000"
    },
    // height: "350px"
  };

  return (
    <>
    <Grid container spacing={2} padding={2}  alignItems={stretch_style}>
      <Grid item xs={9} sx={{height:graphHeight+'px',border:1, borderColor:'primary.main',borderRadius:3,verticalAlign:'center'}} className={item_style} >
        <Graph className={stretch_style}
          graph={data}
          options={options}
          events={events}
        />
      </Grid>
      <Grid item xs={3} sx={{border:1, borderColor:'secondary.main',borderRadius:3,maxHeight:graphHeight+'px',overflow:'scroll'}} className={item_style}>
        <GraphTable className={stretch_style}
          data={graphdata}
          q={q}
        />
      </Grid>
    </Grid>
  </>

  );
}

export default memo(RGV);