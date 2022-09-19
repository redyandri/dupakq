import React from "react";
import { useTable, useSortBy } from 'react-table';
import { Navigate } from 'react-router';




export default function Table({ columns, data }) {
    
  const defaultColumn = {
    width: "auto",
  }

    // Use the useTable Hook to send the columns and data to build the table
    const {
      getTableProps, // table props from react-table
      getTableBodyProps, // table body props from react-table
      headerGroups, // headerGroups, if your table has groupings
      rows, // rows for the table based on the data passed
      prepareRow // Prepare the row (this function needs to be called for each row before getting the row props)
    } = useTable({
      columns,
      data
    },useSortBy);
  
    /* 
      Render the UI for your table
      - react-table doesn't have UI, it's headless. We just need to put the react-table props from the Hooks, and it will do its magic automatically
    */


    
    return (
      <table className="w-full text-md bg-white shadow-md rounded mb-4" {...getTableProps()} style={{ width: '100%' }}>
        <thead>
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map(column => (
                <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                  {column.render("Header")}
                  </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()}>
          {rows.map((row, i) => {
            prepareRow(row);
            return (
              <tr {...row.getRowProps()} >
                {row.cells.map(cell => {
                  return <td {...cell.getCellProps({style: { minWidth: cell.column.minWidth, width: cell.column.width,maxWidth:cell.column.maxWidth }}) } >
                    {cell.render("Cell")}
                    </td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  }