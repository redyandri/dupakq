// import React, { useEffect, useRef } from "react";
// import useResizeAware from "react-resize-aware";
// import PropTypes from "prop-types";
// import Neovis from "neovis.js";// "neovis.js/dist/neovis.js";

// const NeoGraph = (props) => {
//   const {
//     width,
//     height,
//     containerId,
//     backgroundColor,
//     neo4jUri,
//     neo4jUser,
//     neo4jPassword,
//   } = props;

//   const visRef = useRef();

//   useEffect(() => {
//     const config = {
//       container_id: visRef.current.id,
//       server_url: neo4jUri,
//       server_user: neo4jUser,
//       server_password: neo4jPassword,
//       labels: {
//         NODE: {
//           label: "sistem informasi",
//         },
//       },
//       relationships: {
//         MELAKUKAN: {
//           idx: 259,
//           score: 0.11,
//         },
//       },
//       initial_cypher:
//       "match p=(m:NODE)-[]-() where m.label='sistem informasi' return p",
//     };
//     const vis = new Neovis(config);
//     console.log("viz",vis)
//     vis.render();
//   }, [neo4jUri, neo4jUser, neo4jPassword]);

//   return (
//     <div
//       id={containerId}
//       ref={visRef}
//       style={{
//         width: `${width}px`,
//         height: `${height}px`,
//         backgroundColor: `${backgroundColor}`,
//       }}
//     />
//   );
// };

// NeoGraph.defaultProps = {
//   width: 600,
//   height: 600,
//   backgroundColor: "#d3d3d3",
// };

// NeoGraph.propTypes = {
//   width: PropTypes.number.isRequired,
//   height: PropTypes.number.isRequired,
//   containerId: PropTypes.string.isRequired,
//   neo4jUri: PropTypes.string.isRequired,
//   neo4jUser: PropTypes.string.isRequired,
//   neo4jPassword: PropTypes.string.isRequired,
//   backgroundColor: PropTypes.string,
// };

// const ResponsiveNeoGraph = (props) => {
//   const [resizeListener, sizes] = useResizeAware();

//   const side = Math.max(sizes.width, sizes.height) / 2;
//   const neoGraphProps = { ...props, width: side, height: side };
//   return (
//     <div style={{ position: "relative" }}>
//       {resizeListener}
//       <NeoGraph {...neoGraphProps} />
//     </div>
//   );
// };

// ResponsiveNeoGraph.defaultProps = {
//   backgroundColor: "#d3d3d3",
// };

// ResponsiveNeoGraph.propTypes = {
//   containerId: PropTypes.string.isRequired,
//   neo4jUri: PropTypes.string.isRequired,
//   neo4jUser: PropTypes.string.isRequired,
//   neo4jPassword: PropTypes.string.isRequired,
//   backgroundColor: PropTypes.string,
// };

// export { NeoGraph, ResponsiveNeoGraph };