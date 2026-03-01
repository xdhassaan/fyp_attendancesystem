import { useState, useEffect } from 'react';
import { Plus, Pencil, Trash2, Search } from 'lucide-react';
import api, { getErrorMessage } from '../../lib/api';
import { DataTable } from '../../components/ui/DataTable';
import { Modal } from '../../components/ui/Modal';
import { PageHeader } from '../../components/ui/PageHeader';

interface Student {
  id: number;
  registrationNumber: string;
  firstName: string;
  lastName: string;
  email: string;
  phone?: string;
  isActive: boolean;
}

interface StudentFormData {
  registrationNumber: string;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
}

export default function Students() {
  const [students, setStudents] = useState<Student[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<Student | null>(null);
  const [form, setForm] = useState<StudentFormData>({
    registrationNumber: '',
    firstName: '',
    lastName: '',
    email: '',
    phone: ''
  });
  const [saving, setSaving] = useState(false);

  const fetchStudents = async () => {
    setLoading(true);
    try {
      const { data } = await api.get(`/admin/students?page=${page}&limit=20&search=${search}`);
      setStudents(data.data.items);
      setTotalPages(data.data.pagination.totalPages);
      setTotalItems(data.data.pagination.totalItems);
    } catch (error) {
      console.error('Failed to fetch students:', getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStudents();
  }, [page, search]);

  const handleCreate = () => {
    setEditing(null);
    setForm({
      registrationNumber: '',
      firstName: '',
      lastName: '',
      email: '',
      phone: ''
    });
    setModalOpen(true);
  };

  const handleEdit = (student: Student) => {
    setEditing(student);
    setForm({
      registrationNumber: student.registrationNumber,
      firstName: student.firstName,
      lastName: student.lastName,
      email: student.email,
      phone: student.phone || ''
    });
    setModalOpen(true);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this student?')) return;

    try {
      await api.delete(`/admin/students/${id}`);
      fetchStudents();
    } catch (error) {
      alert('Failed to delete student: ' + getErrorMessage(error));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      if (editing) {
        await api.put(`/admin/students/${editing.id}`, form);
      } else {
        await api.post('/admin/students', form);
      }
      setModalOpen(false);
      fetchStudents();
    } catch (error) {
      alert('Failed to save student: ' + getErrorMessage(error));
    } finally {
      setSaving(false);
    }
  };

  const columns = [
    {
      key: 'registrationNumber',
      label: 'Registration #',
      render: (student: Student) => student.registrationNumber
    },
    {
      key: 'name',
      label: 'Name',
      render: (student: Student) => `${student.firstName} ${student.lastName}`
    },
    {
      key: 'email',
      label: 'Email',
      render: (student: Student) => student.email
    },
    {
      key: 'status',
      label: 'Status',
      render: (student: Student) => (
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          student.isActive
            ? 'bg-green-500/20 text-green-400'
            : 'bg-red-500/20 text-red-400'
        }`}>
          {student.isActive ? 'Active' : 'Inactive'}
        </span>
      )
    },
    {
      key: 'actions',
      label: 'Actions',
      render: (student: Student) => (
        <div className="flex gap-2">
          <button
            onClick={() => handleEdit(student)}
            className="p-1 text-maroon-400 hover:text-maroon-300"
          >
            <Pencil size={16} />
          </button>
          <button
            onClick={() => handleDelete(student.id)}
            className="p-1 text-maroon-400 hover:text-maroon-300"
          >
            <Trash2 size={16} />
          </button>
        </div>
      )
    }
  ];

  return (
    <div className="p-6">
      <PageHeader
        title="Students"
        subtitle={`${totalItems} total students`}
      />

      <div className="mb-6 flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white/30" size={20} />
          <input
            type="text"
            placeholder="Search students..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="glass-input w-full pl-10 pr-4 py-2"
          />
        </div>
        <button
          onClick={handleCreate}
          className="btn-gradient flex items-center gap-2 px-4 py-2 rounded-lg"
        >
          <Plus size={20} />
          Add Student
        </button>
      </div>

      <DataTable
        columns={columns}
        data={students}
        loading={loading}
        pagination={{
          page,
          totalPages,
          onPageChange: setPage
        }}
      />

      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editing ? 'Edit Student' : 'Add Student'}
      >
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-white/60 mb-1">
              Registration Number *
            </label>
            <input
              type="text"
              required
              value={form.registrationNumber}
              onChange={(e) => setForm({ ...form, registrationNumber: e.target.value })}
              className="glass-input w-full px-3 py-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-white/60 mb-1">
              First Name *
            </label>
            <input
              type="text"
              required
              value={form.firstName}
              onChange={(e) => setForm({ ...form, firstName: e.target.value })}
              className="glass-input w-full px-3 py-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-white/60 mb-1">
              Last Name *
            </label>
            <input
              type="text"
              required
              value={form.lastName}
              onChange={(e) => setForm({ ...form, lastName: e.target.value })}
              className="glass-input w-full px-3 py-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-white/60 mb-1">
              Email *
            </label>
            <input
              type="email"
              required
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              className="glass-input w-full px-3 py-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-white/60 mb-1">
              Phone
            </label>
            <input
              type="tel"
              value={form.phone}
              onChange={(e) => setForm({ ...form, phone: e.target.value })}
              className="glass-input w-full px-3 py-2"
            />
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={() => setModalOpen(false)}
              className="glass flex-1 px-4 py-2 text-white/60 hover:text-white/80 rounded-lg"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="btn-gradient flex-1 px-4 py-2 rounded-lg disabled:opacity-50"
            >
              {saving ? 'Saving...' : editing ? 'Update' : 'Create'}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
