// import React from "react";
import ReactDOM from "react-dom";
// import Graph from "react-graph-vis";
import Graph from 'react-vis-network-graph';
import React, { useState } from "react";

 
function RGV() {
  // const graph = {
  //   nodes: [
  //     { id: 1, label: "Node 1", title: "node 1 tootip text" },
  //     { id: 2, label: "Node 2", title: "node 2 tootip text" },
  //     { id: 3, label: "Node 3", title: "node 3 tootip text" },
  //     { id: 4, label: "Node 4", title: "node 4 tootip text" },
  //     { id: 5, label: "Node 5", title: "node 5 tootip text" }
  //   ],
  //   edges: [
  //     { from: 1, to: 2,label:'12' },
  //     { from: 1, to: 3,label:'12' },
  //     { from: 2, to: 4,label:'12' },
  //     { from: 2, to: 5,label:'12' }
  //   ]
  // };
  const [state, setState] = useState({
    counter: 5,
    graph: {
      nodes: [
        { id: 1, label: "Node 1", color: "#e04141" },
        { id: 2, label: "Node 2", color: "#e09c41" },
        { id: 3, label: "Node 3", color: "#e0df41" },
        { id: 4, label: "Node 4", color: "#7be041" },
        { id: 5, label: "Node 5", color: "#41e0c9" }
      ],
      edges: [
        { id:12,from: 1, to: 2 ,label:'12'},
        { id:13,from: 1, to: 3 ,label:'13'},
        { id:24,from: 2, to: 4 ,label:'24'},
        { id:25,from: 2, to: 5,label:'25' }
      ]
    },
    events: {
      select: ({ nodes, edges }) => {
        console.log("Selected nodes:"+nodes);
        console.log("Selected edges:"+edges);
      },
      selectEdge:(e)=>{
        var {nodes,edges}=e;
        if (edges.length>=1){
          console.log('selectEdge:'+edges+'\nNodes:'+nodes);
        }
      }
    }
  })
  const { graph, events } = state;
 
  const options = {
    layout: {
      hierarchical: true
    },
    edges: {
      color: "#000000"
    },
    height: "500px"
  };
 
  // const events = {
  //   select: function(event) {
  //     var { nodes, edges } = event;
  //   }
  // };
  return (
    // <Graph
    //   graph={graph}
    //   options={options}
    //   events={events}
    //   getNetwork={network => {
    //     //  if you want access to vis.js network api you can set the state in a parent component using this property
    //   }}
    // />
    <Graph graph={graph} options={options} events={events} style={{ height: "640px" }} />
  );
}
 
export default RGV;


// // import Graph from "react-graph-vis";
// import Graph from 'react-vis-network-graph';
// import React, { useState } from "react";
// import ReactDOM from "react-dom";

// const options = {
//   layout: {
//     hierarchical: false
//   },
//   edges: {
//     color: "#000000"
//   }
// };

// function randomColor() {
//   const red = Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
//   const green = Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
//   const blue = Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
//   return `#${red}${green}${blue}`;
// }

// const myApp = () => {
//   const createNode = (x, y) => {
//     const color = randomColor();
//     setState(({ graph: { nodes, edges }, counter, ...rest }) => {
//       const id = counter + 1;
//       const from = Math.floor(Math.random() * (counter - 1)) + 1;
//       return {
//         graph: {
//           nodes: [
//             ...nodes,
//             { id, label: `Node ${id}`, color, x, y }
//           ],
//           edges: [
//             ...edges,
//             { from, to: id }
//           ]
//         },
//         counter: id,
//         ...rest
//       }
//     });
//   }
//   const [state, setState] = useState({
//     counter: 5,
//     graph: {
//       nodes: [
//         { id: 1, label: "Node 1", color: "#e04141" },
//         { id: 2, label: "Node 2", color: "#e09c41" },
//         { id: 3, label: "Node 3", color: "#e0df41" },
//         { id: 4, label: "Node 4", color: "#7be041" },
//         { id: 5, label: "Node 5", color: "#41e0c9" }
//       ],
//       edges: [
//         { from: 1, to: 2 },
//         { from: 1, to: 3 },
//         { from: 2, to: 4 },
//         { from: 2, to: 5 }
//       ]
//     },
//     events: {
//       select: ({ nodes, edges }) => {
//         console.log("Selected nodes:");
//         console.log(nodes);
//         console.log("Selected edges:");
//         console.log(edges);
//         alert("Selected node: " + nodes);
//       },
//       doubleClick: ({ pointer: { canvas } }) => {
//         createNode(canvas.x, canvas.y);
//       }
//     }
//   })
//   const { graph, events } = state;
//   return (
//     <div>
//       <h1>React graph vis</h1>
//       <p>
//         <a href="https://github.com/crubier/react-graph-vis">Github</a> -{" "}
//         <a href="https://www.npmjs.com/package/react-graph-vis">NPM</a>
//       </p>
//       <p><a href="https://github.com/crubier/react-graph-vis/tree/master/example/src/index.js">Source of this page</a></p>
//       <p>A React component to display beautiful network graphs using vis.js</p>
//       <p>Make sure to visit <a href="http://visjs.org">visjs.org</a> for more info.</p>
//       <p>This package allows to render network graphs using vis.js.</p>
//       <p>Rendered graphs are scrollable, zoomable, retina ready, dynamic</p>
//       <p>In this example, we manage state with react: on double click we create a new node, and on select we display an alert.</p>
//       <Graph graph={graph} options={options} events={events} style={{ height: "640px" }} />
//     </div>
//   );

// }

// export default myApp;