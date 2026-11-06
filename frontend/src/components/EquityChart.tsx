import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface EquityPoint {
  time: string
  equity: number
}

interface EquityChartProps {
  data: EquityPoint[]
}

function EquityChart({ data }: EquityChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient
            id="equityGradient"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop offset="0%" stopColor="#20c997" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#20c997" stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid
          stroke="#243246"
          strokeDasharray="3 5"
          vertical={false}
        />

        <XAxis
          dataKey="time"
          tick={{ fill: '#8492a6', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />

        <YAxis
          domain={['dataMin - 20', 'dataMax + 20']}
          tick={{ fill: '#8492a6', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(value: number) =>
            `₹${(value / 1000).toFixed(1)}k`
          }
          width={58}
        />

        <Tooltip
          contentStyle={{
            background: '#111d2d',
            border: '1px solid #304158',
            borderRadius: 8,
            color: '#e2e8f0',
          }}
          formatter={(value) => [
            `₹${Number(value).toLocaleString('en-IN')}`,
            'Equity',
          ]}
        />

        <Area
          type="monotone"
          dataKey="equity"
          stroke="#20c997"
          strokeWidth={2}
          fill="url(#equityGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default EquityChart
